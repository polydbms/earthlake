"""Tests for benchmark DuckDB persistence module."""

import tempfile
import os

import pytest

import REMSA.benchmark_db as benchmark_db


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Redirect benchmark_db to a temporary DuckDB file for each test."""
    db_file = str(tmp_path / "test_benchmark.duckdb")
    monkeypatch.setattr(benchmark_db, "_DB_PATH", db_file)
    monkeypatch.setattr(benchmark_db, "_con", None)
    yield
    # Close connection after test
    if benchmark_db._con is not None:
        benchmark_db._con.close()
        benchmark_db._con = None


def _make_job(
    job_id="job-1",
    model_variant="prithvi_100m",
    dataset="eurosat",
    mode="test",
    status="completed",
    metrics=None,
    duration_seconds=42.5,
    error=None,
    config_path="/configs/test.yaml",
    submitted_at="2025-01-01T00:00:00",
    completed_at="2025-01-01T00:01:00",
):
    return {
        "job_id": job_id,
        "model_variant": model_variant,
        "dataset": dataset,
        "mode": mode,
        "status": status,
        "metrics": metrics,
        "duration_seconds": duration_seconds,
        "error": error,
        "config_path": config_path,
        "submitted_at": submitted_at,
        "completed_at": completed_at,
    }


# ---------------------------------------------------------------------------
# init_benchmark_table
# ---------------------------------------------------------------------------

def test_init_benchmark_table_idempotent():
    """Calling init twice should not raise."""
    benchmark_db.init_benchmark_table()
    benchmark_db.init_benchmark_table()


# ---------------------------------------------------------------------------
# upsert + query round-trip
# ---------------------------------------------------------------------------

def test_insert_completed_job_with_metrics():
    benchmark_db.init_benchmark_table()
    job = _make_job(metrics={"accuracy": 0.95, "f1": 0.90, "miou": 0.85, "loss": 0.12})
    benchmark_db.upsert_benchmark_result(job)

    rows = benchmark_db.query_benchmark_results()
    assert len(rows) == 1
    row = rows[0]
    assert row["job_id"] == "job-1"
    assert row["model_variant"] == "prithvi_100m"
    assert row["dataset"] == "eurosat"
    assert row["status"] == "completed"
    assert row["accuracy"] == pytest.approx(0.95)
    assert row["f1"] == pytest.approx(0.90)
    assert row["miou"] == pytest.approx(0.85)
    assert row["loss"] == pytest.approx(0.12)
    assert row["duration_seconds"] == pytest.approx(42.5)
    assert row["config_path"] == "/configs/test.yaml"


def test_insert_failed_job_with_no_metrics():
    benchmark_db.init_benchmark_table()
    job = _make_job(
        job_id="job-fail",
        status="failed",
        metrics=None,
        error="GPU OOM",
        config_path=None,
        duration_seconds=None,
    )
    benchmark_db.upsert_benchmark_result(job)

    rows = benchmark_db.query_benchmark_results()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "failed"
    assert row["accuracy"] is None
    assert row["f1"] is None
    assert row["miou"] is None
    assert row["loss"] is None
    assert row["error_message"] == "GPU OOM"


def test_upsert_overwrites_existing():
    benchmark_db.init_benchmark_table()
    job_v1 = _make_job(status="running", metrics=None)
    benchmark_db.upsert_benchmark_result(job_v1)

    job_v2 = _make_job(
        status="completed",
        metrics={"accuracy": 0.99},
    )
    benchmark_db.upsert_benchmark_result(job_v2)

    rows = benchmark_db.query_benchmark_results()
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["accuracy"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# query filters
# ---------------------------------------------------------------------------

def test_query_filter_by_model_variant():
    benchmark_db.init_benchmark_table()
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j1", model_variant="prithvi_100m"))
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j2", model_variant="satmae"))

    rows = benchmark_db.query_benchmark_results(model_variant="prithvi_100m")
    assert len(rows) == 1
    assert rows[0]["model_variant"] == "prithvi_100m"


def test_query_filter_by_dataset():
    benchmark_db.init_benchmark_table()
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j1", dataset="eurosat"))
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j2", dataset="bigearthnet"))

    rows = benchmark_db.query_benchmark_results(dataset="bigearthnet")
    assert len(rows) == 1
    assert rows[0]["dataset"] == "bigearthnet"


def test_query_filter_combined():
    benchmark_db.init_benchmark_table()
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j1", model_variant="prithvi_100m", dataset="eurosat"))
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j2", model_variant="prithvi_100m", dataset="bigearthnet"))
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j3", model_variant="satmae", dataset="eurosat"))

    rows = benchmark_db.query_benchmark_results(model_variant="prithvi_100m", dataset="eurosat")
    assert len(rows) == 1
    assert rows[0]["job_id"] == "j1"


# ---------------------------------------------------------------------------
# load_completed_jobs
# ---------------------------------------------------------------------------

def test_load_completed_jobs_returns_only_terminal():
    benchmark_db.init_benchmark_table()
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j-done", status="completed"))
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j-fail", status="failed", error="err"))
    benchmark_db.upsert_benchmark_result(_make_job(job_id="j-run", status="running"))

    jobs = benchmark_db.load_completed_jobs()
    assert "j-done" in jobs
    assert "j-fail" in jobs
    assert "j-run" not in jobs


def test_load_completed_jobs_dict_structure():
    benchmark_db.init_benchmark_table()
    benchmark_db.upsert_benchmark_result(
        _make_job(
            job_id="j1",
            metrics={"accuracy": 0.9, "f1": 0.8},
        )
    )

    jobs = benchmark_db.load_completed_jobs()
    job = jobs["j1"]
    assert job["job_id"] == "j1"
    assert job["model_variant"] == "prithvi_100m"
    assert job["dataset"] == "eurosat"
    assert job["mode"] == "test"
    assert job["status"] == "completed"
    assert job["metrics"] == {"accuracy": pytest.approx(0.9), "f1": pytest.approx(0.8)}
    assert "error" in job
    assert "config_path" in job
