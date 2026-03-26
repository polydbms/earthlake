"""Tests for the benchmark REST API.

Uses FastAPI TestClient — no server, no langchain dependency needed.
Run directly: python REMSA/tests/test_benchmark_api.py
"""

import os
import tempfile
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Redirect benchmark_db to a temp file before importing the router,
# so that module-level init in benchmark_api works in test environments.
import REMSA.benchmark_db as _bdb

_tmp_dir = tempfile.mkdtemp()
_bdb._DB_PATH = os.path.join(_tmp_dir, "test_benchmark.duckdb")
_bdb._con = None
_bdb.init_benchmark_table()

from REMSA.benchmark_api import router

# Build a minimal app that only mounts the benchmark router
_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


# ---------------------------------------------------------------------------
# 1. GET /benchmark/models
# ---------------------------------------------------------------------------
def test_list_models():
    resp = client.get("/benchmark/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    first = data[0]
    for key in ("family", "variant", "description", "min_gpu_memory"):
        assert key in first, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 2. GET /benchmark/datasets
# ---------------------------------------------------------------------------
def test_list_datasets():
    resp = client.get("/benchmark/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 6
    first = data[0]
    for key in ("name", "task", "num_classes", "size", "modality", "resolution"):
        assert key in first, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 3. GET /benchmark/supports/prithvi_100m  → supported: true
# ---------------------------------------------------------------------------
def test_supports_known_model():
    resp = client.get("/benchmark/supports/prithvi_100m")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "prithvi_100m"
    assert data["supported"] is True


# ---------------------------------------------------------------------------
# 4. GET /benchmark/supports/nonexistent  → supported: false
# ---------------------------------------------------------------------------
def test_supports_unknown_model():
    resp = client.get("/benchmark/supports/nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "nonexistent"
    assert data["supported"] is False


# ---------------------------------------------------------------------------
# 5. GET /benchmark/gpu?model_variant=prithvi_100m
# ---------------------------------------------------------------------------
def test_gpu_check():
    resp = client.get("/benchmark/gpu", params={"model_variant": "prithvi_100m"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_variant"] == "prithvi_100m"
    assert "meets_requirements" in data
    assert "required_memory" in data


# ---------------------------------------------------------------------------
# 6. POST /benchmark/run  → 202 with pending job
# ---------------------------------------------------------------------------
def test_submit_benchmark():
    resp = client.post("/benchmark/run", json={
        "model_variant": "prithvi_100m",
        "dataset": "eurosat",
    })
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert data["model_variant"] == "prithvi_100m"
    assert data["dataset"] == "eurosat"


# ---------------------------------------------------------------------------
# 7. Async lifecycle: submit → poll → failed (no terratorch)
# ---------------------------------------------------------------------------
def test_job_lifecycle():
    resp = client.post("/benchmark/run", json={
        "model_variant": "prithvi_100m",
        "dataset": "eurosat",
    })
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Poll until the background task finishes (should fail fast)
    for _ in range(20):
        resp = client.get(f"/benchmark/jobs/{job_id}")
        assert resp.status_code == 200
        status = resp.json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.5)

    data = resp.json()
    assert data["status"] == "failed"
    assert data["error"] is not None


# ---------------------------------------------------------------------------
# 8. GET /benchmark/jobs  → list with ≥1 job
# ---------------------------------------------------------------------------
def test_list_jobs():
    # Ensure at least one job exists (from earlier tests)
    resp = client.get("/benchmark/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


# ---------------------------------------------------------------------------
# 9. GET /benchmark/jobs?status=failed  → filtered list
# ---------------------------------------------------------------------------
def test_list_jobs_filtered():
    resp = client.get("/benchmark/jobs", params={"status": "failed"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    for job in data:
        assert job["status"] == "failed"


# ---------------------------------------------------------------------------
# 10. GET /benchmark/jobs/nonexistent  → 404
# ---------------------------------------------------------------------------
def test_get_job_not_found():
    resp = client.get("/benchmark/jobs/nonexistent-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 11. POST /benchmark/run with invalid batch_size  → 422
# ---------------------------------------------------------------------------
def test_validation_error():
    resp = client.post("/benchmark/run", json={
        "model_variant": "prithvi_100m",
        "dataset": "eurosat",
        "batch_size": -5,
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 12. GET /benchmark/results  → list
# ---------------------------------------------------------------------------
def test_get_results():
    resp = client.get("/benchmark/results")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


# ---------------------------------------------------------------------------
# 13. GET /benchmark/results?model_variant=...  → filtered list
# ---------------------------------------------------------------------------
def test_get_results_filtered():
    resp = client.get("/benchmark/results", params={"model_variant": "nonexistent_model"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    # With a nonexistent model, should return empty
    assert len(data) == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_list_models,
        test_list_datasets,
        test_supports_known_model,
        test_supports_unknown_model,
        test_gpu_check,
        test_submit_benchmark,
        test_job_lifecycle,
        test_list_jobs,
        test_list_jobs_filtered,
        test_get_job_not_found,
        test_validation_error,
        test_get_results,
        test_get_results_filtered,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        raise SystemExit(1)
