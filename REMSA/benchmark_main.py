"""Standalone FastAPI entry point for the benchmark service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from REMSA.benchmark_api import router as benchmark_router

app = FastAPI(title="REMSA Benchmark Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(benchmark_router)


@app.get("/health")
def health():
    gpu_status = "unknown"
    try:
        import torch

        gpu_status = "available" if torch.cuda.is_available() else "unavailable"
    except ImportError:
        gpu_status = "torch_not_installed"
    return {"status": "ok", "gpu": gpu_status}
