from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid

from .run import run_extraction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...)):
    tmp = Path("/tmp") / f"{uuid.uuid4()}.pdf"
    with tmp.open("wb") as f:
        f.write(await file.read())

    result = run_extraction(str(tmp))
    return JSONResponse(result)