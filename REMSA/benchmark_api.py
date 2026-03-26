"""Benchmark Service REST API with async job execution."""

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
import queue
import subprocess
import shutil

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path

from REMSA.benchmark_db import (
    init_benchmark_table,
    load_completed_jobs,
    query_benchmark_results,
    upsert_benchmark_result,
    delete_benchmark_result,
)
from REMSA.benchmark_runner import BenchmarkRunner

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BenchmarkRunRequest(BaseModel):
    model_variant: str
    dataset: str
    mode: Literal["test", "linear_probe", "finetune"] = "test"
    batch_size: int = Field(default=8, ge=1, le=256)
    max_epochs: int = Field(default=10, ge=1, le=1000)
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    num_workers: int = Field(default=4, ge=0, le=32)
    checkpoint_path: Optional[str] = None


class ModelInfo(BaseModel):
    family: str
    variant: str
    description: str
    min_gpu_memory: int


class DatasetInfo(BaseModel):
    name: str
    task: str
    num_classes: int
    size: str
    modality: str
    resolution: str


class SupportsResponse(BaseModel):
    model_name: str
    supported: bool


class GpuCheckResponse(BaseModel):
    model_variant: str
    meets_requirements: bool
    required_memory: int
    available_memory: Optional[float] = None
    error: Optional[str] = None


class BenchmarkJobResponse(BaseModel):
    job_id: str
    status: str
    model_variant: str
    dataset: str
    mode: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    has_checkpoint: bool = False
    checkpoint_path: Optional[str] = None

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


_runner = BenchmarkRunner()
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_job_queue: "queue.Queue[tuple[str, BenchmarkRunRequest]]" = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False
_running_processes: Dict[str, subprocess.Popen] = {}

# Hydrate in-memory state from DuckDB on startup
try:
    init_benchmark_table()
    _jobs.update(load_completed_jobs())
except Exception:
    logger.exception("Failed to initialize benchmark DB; continuing in-memory only")

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

def _worker_loop():
    while True:
        job_id, request = _job_queue.get()

        try:
            with _jobs_lock:
                job = _jobs.get(job_id)

                # skip cancelled or deleted jobs
                if not job or job["status"] in ["cancelled", "deleted"]:
                    _job_queue.task_done()
                    continue

            _run_benchmark_sync(job_id, request)

        finally:
            _job_queue.task_done()

def _ensure_worker_started():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=_worker_loop, daemon=True)
            t.start()
            _worker_started = True

def _run_benchmark_sync(job_id: str, request: BenchmarkRunRequest):
    now = datetime.now(timezone.utc).isoformat()

    with _jobs_lock:
        if _jobs[job_id]["status"] in ["cancelled", "deleted"]:
            return
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = now

    def register_process(p: subprocess.Popen):
        _running_processes[job_id] = p

    try:
        result = _runner.run_benchmark(
            model_variant=request.model_variant,
            dataset=request.dataset,
            job_id=job_id,
            mode=request.mode,
            batch_size=request.batch_size,
            max_epochs=request.max_epochs,
            learning_rate=request.learning_rate,
            num_workers=request.num_workers,
            on_process_started=register_process,
        )
        completed = datetime.now(timezone.utc).isoformat()
        with _jobs_lock:
            if _jobs[job_id]["status"] != "cancelled":
                _jobs[job_id]["status"] = "completed" if result.status == "success" else "failed"
                _jobs[job_id]["completed_at"] = completed
                _jobs[job_id]["duration_seconds"] = result.duration_seconds
                _jobs[job_id]["metrics"] = result.metrics
                _jobs[job_id]["error"] = result.error_message
                _jobs[job_id]["config_path"] = getattr(result, "config_path", None)
                _jobs[job_id]["mode"] = request.mode
                try:
                    upsert_benchmark_result(_jobs[job_id])
                except Exception:
                    logger.exception("Failed to persist benchmark result for job %s", job_id)

    except Exception as exc:
        completed = datetime.now(timezone.utc).isoformat()
        with _jobs_lock:
            if _jobs[job_id]["status"] != "cancelled":
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["completed_at"] = completed
                _jobs[job_id]["error"] = str(exc)

                try:
                    upsert_benchmark_result(_jobs[job_id])
                except Exception:
                    logger.exception("Failed to persist benchmark result for job %s", job_id)

    finally:
        _running_processes.pop(job_id, None)
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/models", response_model=List[ModelInfo])
def list_models():
    return _runner.get_available_models()


@router.get("/datasets", response_model=List[DatasetInfo])
def list_datasets():
    return _runner.get_available_datasets()


@router.get("/supports/{model_name}", response_model=SupportsResponse)
def check_support(model_name: str):
    return SupportsResponse(
        model_name=model_name,
        supported=_runner.supports_benchmark(model_name),
    )


@router.post("/run", response_model=BenchmarkJobResponse, status_code=202)
def submit_benchmark(request: BenchmarkRunRequest):
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "queued",
        "model_variant": request.model_variant,
        "dataset": request.dataset,
        "mode": request.mode,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    with _jobs_lock:
        _jobs[job_id] = job

    response = BenchmarkJobResponse(**job)
    _ensure_worker_started()
    _job_queue.put((job_id, request))

    return response


@router.get("/jobs/{job_id}", response_model=BenchmarkJobResponse)
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return BenchmarkJobResponse(**job)


@router.get("/jobs", response_model=List[BenchmarkJobResponse])
def list_jobs(status: Optional[str] = Query(default=None)):
    base_results = Path("benchmark_results/results")

    with _jobs_lock:
        jobs = list(_jobs.values())

    if status:
        jobs = [j for j in jobs if j["status"] == status]

    enriched = []

    for j in jobs:
        job_id = j["job_id"]
        ckpt = base_results / job_id / "checkpoint.ckpt"

        enriched.append(
            BenchmarkJobResponse(
                **j,
                has_checkpoint=ckpt.exists(),
                checkpoint_path=str(ckpt) if ckpt.exists() else None,
            )
        )

    return enriched


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):

    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "queued":
        job["status"] = "cancelled"

        try:
            upsert_benchmark_result(job)
        except Exception:
            logger.exception("Failed to persist cancelled job")

        return {"status": "cancelled"}

    process = _running_processes.get(job_id)

    if process:
        process.terminate()

        with _jobs_lock:
            job["status"] = "cancelled"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()

        try:
            upsert_benchmark_result(job)
        except Exception:
            logger.exception("Failed to persist cancelled job")

        return {"status": "cancelled"}

    raise HTTPException(status_code=400, detail="Job cannot be cancelled")

@router.get("/results")
def get_results(
    model_variant: Optional[str] = Query(default=None),
    dataset: Optional[str] = Query(default=None),
):
    """Return persisted benchmark results with optional filters."""
    return query_benchmark_results(model_variant=model_variant, dataset=dataset)


@router.get("/gpu", response_model=GpuCheckResponse)
def check_gpu(model_variant: str = Query(...)):
    result = _runner.check_gpu_requirements(model_variant)
    return GpuCheckResponse(model_variant=model_variant, **result)

@router.get("/{job_id}/metrics")
def get_job_metrics(job_id: str):
    job_dir = Path(f"/app/benchmark_results/results/{job_id}")
    metrics_file = job_dir / "lightning_logs" / "version_0" / "metrics.csv"

    if not metrics_file.exists():
        return {"task": None, "checkpoints": []}

    df = pd.read_csv(metrics_file)

    # Keep only rows with valid epoch
    df = df[df["epoch"].notna()]
    if df.empty:
        return {"task": None, "checkpoints": []}

    df["epoch"] = df["epoch"].astype(int)

    # Sort safely
    if "step" in df.columns:
        df = df.sort_values(["epoch", "step"])
    else:
        df = df.sort_values(["epoch"])

    # Identify metric columns
    train_metric_cols = [c for c in df.columns if c.startswith("train/")]
    val_metric_cols = [c for c in df.columns if c.startswith("val/")]

    # Split (only epoch + metrics, no step)
    train_df = df[["epoch"] + train_metric_cols].dropna(
        subset=train_metric_cols, how="all"
    )
    val_df = df[["epoch"] + val_metric_cols].dropna(
        subset=val_metric_cols, how="all"
    )

    if val_df.empty:
        return {"task": None, "checkpoints": []}

    # Aggregate last values per epoch
    train_epoch = (
        train_df.groupby("epoch").last()
        if not train_df.empty
        else pd.DataFrame()
    )
    val_epoch = val_df.groupby("epoch").last()

    # Merge safely
    if train_epoch.empty:
        df_epoch = val_epoch.reset_index()
    else:
        df_epoch = train_epoch.join(val_epoch, how="outer").reset_index()

    def safe_float(x):
        return None if pd.isna(x) else float(x)

    # Detect task
    if "val/mIoU" in df.columns:
        task = "segmentation"
        checkpoints = [
            {
                "epoch": int(row["epoch"]),
                "train_accuracy": safe_float(row.get("train/Accuracy")),
                "train_miou": safe_float(row.get("train/mIoU")),
                "train_loss": safe_float(row.get("train/loss")), # Added train_loss
                "val_accuracy": safe_float(row.get("val/Accuracy")),
                "val_miou": safe_float(row.get("val/mIoU")),
                "val_loss": safe_float(row.get("val/loss")),
            }
            for _, row in df_epoch.iterrows()
        ]
    else:
        task = "classification"
        checkpoints = [
            {
                "epoch": int(row["epoch"]),
                "train_accuracy": safe_float(row.get("train/Accuracy")),
                "train_f1": safe_float(row.get("train/F1_Score")),
                "train_loss": safe_float(row.get("train/loss")), # Added train_loss
                "val_accuracy": safe_float(row.get("val/Accuracy")),
                "val_f1": safe_float(row.get("val/F1_Score")),
                "val_loss": safe_float(row.get("val/loss")),
            }
            for _, row in df_epoch.iterrows()
        ]

    return {
        "task": task,
        "checkpoints": checkpoints,
    }


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Guardrail: Don't delete active jobs
    if job["status"] in ["queued", "running"]:
        raise HTTPException(status_code=400, detail="Cannot delete an active job. Cancel it first.")

    # 1. Remove from in-memory state
    with _jobs_lock:
        _jobs.pop(job_id, None)

    # 2. Remove from DuckDB
    try:
        delete_benchmark_result(job_id)
    except Exception:
        logger.exception("Failed to delete benchmark result from DB for job %s", job_id)

    # 3. Delete the corresponding folder and its contents
    job_dir = Path(f"benchmark_results/results/{job_id}")
    if job_dir.exists() and job_dir.is_dir():
        try:
            shutil.rmtree(job_dir)
        except Exception as e:
            logger.error(f"Failed to delete directory {job_dir}: {e}")

    return {"status": "deleted"}