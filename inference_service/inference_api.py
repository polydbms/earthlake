from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import shutil

from deploy_service import deploy, get_status
from classification_inference import run as run_classification
from segmentation_inference import run as run_segmentation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/deploy/{job_id}/{task}")
def deploy_model(job_id: str, task: str):
    if task not in ["classification", "segmentation"]:
        raise HTTPException(status_code=400, detail="Invalid task")
    return deploy(job_id, task)

@app.get("/inference/status")
def inference_status():
    return get_status()

@app.post("/inference/predict")
def predict(file: UploadFile = File(...)):
    status = get_status()

    if not status["job_id"]:
        raise HTTPException(status_code=400, detail="No model deployed")

    tmp = Path("tmp")
    tmp.mkdir(exist_ok=True)
    tif_path = tmp / f"{uuid.uuid4()}.tif"

    with tif_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if status["task"] == "classification":
        return run_classification(status["job_id"], tif_path)
    else:
        png_path = tmp / f"{uuid.uuid4()}.png"
        run_segmentation(status["job_id"], tif_path, png_path)
        return FileResponse(png_path, media_type="image/png")