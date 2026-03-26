from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import logging
import traceback
import sys
from time import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("sql_service")

# catch uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.critical("UNCAUGHT EXCEPTION")
    logger.critical("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = handle_exception

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Opening DuckDB database at /model_metadata/model_metadata.duckdb")

db = duckdb.connect(
    "/model_metadata/model_metadata.duckdb",
    read_only=True
)

logger.info("DuckDB connection established")


@app.middleware("http")
async def log_requests(request: Request, call_next):

    start = time()

    logger.info("Request started: %s %s", request.method, request.url.path)

    try:
        response = await call_next(request)
    except Exception:
        logger.error("Unhandled request error")
        logger.error(traceback.format_exc())
        raise

    duration = time() - start

    logger.info(
        "Request completed: %s %s | %s | %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


@app.post("/sql")
def exec_sql(payload: dict):

    sql = payload.get("query", "")

    logger.info("Incoming SQL request")

    if not sql:
        logger.warning("Request missing SQL query")
        raise HTTPException(400, "missing query")

    logger.info("SQL query: %s", sql)

    sql_upper = sql.lstrip().upper()

    if not sql_upper.startswith("SELECT"):
        raise HTTPException(403, "only SELECT queries allowed")

    cleaned = sql.strip()

    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    if ";" in cleaned:
        raise HTTPException(400, "multiple statements not allowed")

    try:

        logger.info("Opening DuckDB connection")

        with duckdb.connect(
            "/model_metadata/model_metadata.duckdb",
            read_only=True
        ) as db:

            logger.info("Executing query")

            result = db.execute(sql)

            rows = result.fetchall()
            cols = [c[0] for c in result.description]

        logger.info("Query returned %d rows", len(rows))

        return [dict(zip(cols, row)) for row in rows]

    except Exception as e:

        logger.error("Query execution failed")
        logger.error(traceback.format_exc())

        raise HTTPException(400, str(e))