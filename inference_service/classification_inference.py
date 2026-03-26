import torch
import rasterio
import numpy as np
import torch.nn.functional as F
import yaml
from pathlib import Path
from functools import lru_cache
from terratorch.tasks import ClassificationTask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_model(job_id: str):
    job_dir = Path(f"/app/benchmark_results/results/{job_id}")

    deploy_cfg = job_dir / "lightning_logs" / "version_0" / "config_deploy.yaml"
    with open(deploy_cfg) as f:
        cfg = yaml.safe_load(f)

    ckpt = sorted(job_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)[-1]

    model = ClassificationTask.load_from_checkpoint(str(ckpt))
    model.eval()
    model.to(DEVICE)

    bands = cfg["data"]["init_args"]["bands"]
    class_names = cfg["data"]["init_args"].get("class_names")

    if not class_names:
        num_classes = cfg["model"]["init_args"].get("num_classes", 10)
        class_names = [f"class_{i}" for i in range(num_classes)]

    artifact = {
        "bands": bands,
        "image_size": 224,
        "class_names": class_names,
    }

    return model, artifact


def run(job_id: str, tif_path: Path):
    model, artifact = load_model(job_id)

    with rasterio.open(tif_path) as src:
        full = src.read()

    bands = artifact["bands"]

    if full.shape[0] == len(bands):
        x_np = full
    else:
        S2_INDEX = {
            "B01": 0, "B02": 1, "B03": 2, "B04": 3,
            "B05": 4, "B06": 5, "B07": 6, "B08": 7,
            "B8A": 8, "B09": 9, "B10": 10,
            "B11": 11, "B12": 12,
        }
        x_np = np.stack([full[S2_INDEX[b]] for b in bands], axis=0)

    x = torch.from_numpy(x_np).float().unsqueeze(0)

    x = F.interpolate(
        x,
        size=(artifact["image_size"], artifact["image_size"]),
        mode="bilinear",
        align_corners=False,
    ).to(DEVICE)

    with torch.inference_mode():
        logits = model(x)
        logits = logits.output if hasattr(logits, "output") else logits
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

    probs_list = probs.squeeze().cpu().tolist()
    pred_index = int(pred.item())

    return {
        "predicted_index": pred_index,
        "predicted_label": artifact["class_names"][pred_index],
        "classes": [
            {
                "index": i,
                "label": artifact["class_names"][i],
                "probability": probs_list[i],
            }
            for i in range(len(probs_list))
        ],
    }