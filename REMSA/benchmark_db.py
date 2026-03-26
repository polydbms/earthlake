"""DuckDB persistence for benchmark results.

Stores benchmark job results in the shared model_metadata DuckDB database
so they survive service restarts and are queryable from MetadataExplorer.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)

_DB_PATH = "/model_metadata/benchmark_results.duckdb"
_db_lock = threading.Lock()
_con: Optional[duckdb.DuckDBPyConnection] = None


def _get_connection() -> duckdb.DuckDBPyConnection:
    """Return a lazy singleton DuckDB connection."""
    global _con
    if _con is None:
        _con = duckdb.connect(_DB_PATH)
    return _con


def init_benchmark_table() -> None:
    """Create the benchmark_results table if it doesn't exist. Idempotent."""
    with _db_lock:
        con = _get_connection()
        con.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                job_id          VARCHAR PRIMARY KEY,
                model_variant   VARCHAR NOT NULL,
                dataset         VARCHAR NOT NULL,
                mode            VARCHAR NOT NULL,
                status          VARCHAR NOT NULL,
                epochs          INTEGER,
                accuracy        DOUBLE,
                f1              DOUBLE,
                miou            DOUBLE,
                loss            DOUBLE,
                duration_seconds DOUBLE,
                error_message   VARCHAR,
                config_path     VARCHAR,
                submitted_at    TIMESTAMP,
                completed_at    TIMESTAMP,
                best_checkpoint_path VARCHAR
            )
        """)
        # Migrate existing databases that predate this column
        con.execute("""
            ALTER TABLE benchmark_results
            ADD COLUMN IF NOT EXISTS epochs INTEGER
        """)


def upsert_benchmark_result(job: dict) -> None:
    """Insert or replace a benchmark result row from a job dict."""
    metrics = job.get("metrics") or {}
    with _db_lock:
        con = _get_connection()
        con.execute(
            """
            INSERT OR REPLACE INTO benchmark_results (
                job_id, model_variant, dataset, mode, status, epochs,
                accuracy, f1, miou, loss,
                duration_seconds, error_message, config_path,
                submitted_at, completed_at, best_checkpoint_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                job["job_id"],
                job["model_variant"],
                job["dataset"],
                job.get("mode", "test"),
                job["status"],
                job.get("epochs"),
                metrics.get("accuracy"),
                metrics.get("f1"),
                metrics.get("miou"),
                metrics.get("loss"),
                job.get("duration_seconds"),
                job.get("error"),
                job.get("config_path"),
                job.get("submitted_at"),
                job.get("completed_at"),
                job.get("best_checkpoint_path"),
            ],
        )


def query_benchmark_results(
    model_variant: Optional[str] = None,
    dataset: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query benchmark results with optional filters.

    Returns rows ordered by completed_at DESC NULLS LAST.
    """
    clauses: List[str] = []
    params: List[Any] = []

    if model_variant is not None:
        clauses.append("model_variant = ?")
        params.append(model_variant)
    if dataset is not None:
        clauses.append("dataset = ?")
        params.append(dataset)

    where = ""
    if clauses:
        where = "WHERE " + " AND ".join(clauses)

    sql = f"SELECT * FROM benchmark_results {where} ORDER BY completed_at DESC NULLS LAST"

    with _db_lock:
        con = _get_connection()
        result = con.execute(sql, params)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

    return [dict(zip(columns, row)) for row in rows]


def load_completed_jobs() -> Dict[str, Dict[str, Any]]:
    """Load completed/failed jobs from DuckDB into the in-memory _jobs format.

    Returns a dict keyed by job_id matching the structure used in benchmark_api._jobs.
    """
    with _db_lock:
        con = _get_connection()
        result = con.execute(
            "SELECT * FROM benchmark_results WHERE status IN ('completed', 'failed')"
        )
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

    jobs: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_dict = dict(zip(columns, row))

        # Reconstruct metrics sub-dict from individual columns
        metrics: Optional[Dict[str, float]] = None
        metric_keys = ("accuracy", "f1", "miou", "loss")
        metric_values = {k: row_dict[k] for k in metric_keys if row_dict.get(k) is not None}
        if metric_values:
            metrics = metric_values

        job_id = row_dict["job_id"]
        jobs[job_id] = {
            "job_id": job_id,
            "model_variant": row_dict["model_variant"],
            "dataset": row_dict["dataset"],
            "mode": row_dict["mode"],
            "status": row_dict["status"],
            "epochs": row_dict.get("epochs"),
            "started_at": row_dict.get("submitted_at"),
            "completed_at": row_dict.get("completed_at"),
            "duration_seconds": row_dict.get("duration_seconds"),
            "metrics": metrics,
            "error": row_dict.get("error_message"),
            "config_path": row_dict.get("config_path"),
            "submitted_at": row_dict.get("submitted_at"),
            "best_checkpoint_path": row_dict.get("best_checkpoint_path"),
        }

    return jobs


def delete_benchmark_result(job_id: str) -> None:
    """Delete a benchmark result from DuckDB."""
    with _db_lock:
        con = _get_connection()
        con.execute(
            "DELETE FROM benchmark_results WHERE job_id = ?",
            [job_id]
        )