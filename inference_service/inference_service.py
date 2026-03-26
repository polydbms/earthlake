import torch
import rasterio
import numpy as np
import torch.nn.functional as F
import yaml
from pathlib import Path
from functools import lru_cache
from terratorch.tasks import ClassificationTask
from terratorch.tasks import SemanticSegmentationTask
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFICATION_JOB_DIR = Path("/app/benchmark_results/results/51b88894-do-not-delete")
SEGMENTATION_JOB_DIR = Path("/app/benchmark_results/results/44b0522b-do-not-delete")


def find_latest_job_dir(base_dir=Path("benchmark_results/results")):
    job_dirs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not job_dirs:
        raise RuntimeError("No benchmark results found")
    return job_dirs[-1]


def load_classification_artifact():
    if not CLASSIFICATION_JOB_DIR.exists():
        raise RuntimeError(f"Job dir not found: {CLASSIFICATION_JOB_DIR}")

    deploy_cfg = CLASSIFICATION_JOB_DIR / "lightning_logs" / "version_0" / "config_deploy.yaml"
    if not deploy_cfg.exists():
        raise RuntimeError(f"Deploy yaml not found: {deploy_cfg}")

    ckpt_files = list(CLASSIFICATION_JOB_DIR.rglob("*.ckpt"))
    if not ckpt_files:
        raise RuntimeError(f"No checkpoint found in {CLASSIFICATION_JOB_DIR}")
    ckpt = ckpt_files[0]

    with open(deploy_cfg) as f:
        cfg = yaml.safe_load(f)

    bands = cfg["data"]["init_args"]["bands"]
    image_size = 224

    # Try to extract class names safely
    class_names = (
        cfg.get("data", {})
           .get("init_args", {})
           .get("class_names")
    )

    # Fallback to numeric class names if not defined
    if class_names is None:
        num_classes = (
            cfg.get("model", {})
               .get("init_args", {})
               .get("num_classes")
        )
        if num_classes is None:
            num_classes = 10  # safe fallback
        class_names = [f"class_{i}" for i in range(num_classes)]

    return {
        "ckpt": ckpt,
        "bands": bands,
        "image_size": image_size,
        "class_names": class_names,
    }

def load_segmentation_artifact():
    if not SEGMENTATION_JOB_DIR.exists():
        raise RuntimeError(f"Job dir not found: {SEGMENTATION_JOB_DIR}")

    deploy_cfg = SEGMENTATION_JOB_DIR / "lightning_logs" / "version_0" / "config_deploy.yaml"
    if not deploy_cfg.exists():
        raise RuntimeError(f"Deploy yaml not found: {deploy_cfg}")

    ckpt_files = sorted(
        SEGMENTATION_JOB_DIR.rglob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )

    # Avoid state_dict-only checkpoints
    ckpt_files = [p for p in ckpt_files if not p.name.endswith("_state_dict.ckpt")]
    if not ckpt_files:
        raise RuntimeError(f"No valid checkpoint found in {SEGMENTATION_JOB_DIR}")

    ckpt = ckpt_files[-1]

    with open(deploy_cfg) as f:
        cfg = yaml.safe_load(f)

    init_args = cfg["data"]["init_args"]

    bands = init_args.get("output_bands") or init_args.get("dataset_bands")
    if not bands:
        raise RuntimeError("No bands found in segmentation deploy config")

    image_size = 224

    return {
        "ckpt": ckpt,
        "bands": bands,
        "image_size": image_size,
    }

@lru_cache(maxsize=1)
def get_model():
    #job_dir = find_latest_job_dir()
    artifact = load_classification_artifact()

    model = ClassificationTask.load_from_checkpoint(
        str(artifact["ckpt"])
    )
    model.eval()
    model.to(DEVICE)

    return model, artifact

@lru_cache(maxsize=1)
def get_segmentation_model():
    artifact = load_segmentation_artifact()

    model = SemanticSegmentationTask.load_from_checkpoint(
        str(artifact["ckpt"])
    )
    model.eval()
    model.to(DEVICE)

    return model, artifact


def run_classification(tif_path: Path):
    print("Running classification task...")
    model, artifact = get_model()

    bands = artifact["bands"]
    image_size = artifact["image_size"]

    with rasterio.open(tif_path) as src:
        full = src.read()  # shape: (bands, H, W)

    # If TIFF already has correct band count and order
    if full.shape[0] == len(bands):
        x_np = full
    else:
        # assume Sentinel-2 full product
        S2_INDEX = {
            "B01": 0, "B02": 1, "B03": 2, "B04": 3,
            "B05": 4, "B06": 5, "B07": 6, "B08": 7,
            "B8A": 8, "B09": 9, "B10": 10,
            "B11": 11, "B12": 12,
        }
        x_np = np.stack([full[S2_INDEX[b]] for b in bands], axis=0)

    x = torch.from_numpy(x_np).float().unsqueeze(0)

    # Resize to training resolution
    x = F.interpolate(
        x,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).to(DEVICE)

    with torch.inference_mode():
        out = model(x)
        logits = out.output if hasattr(out, "output") else out
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)


    probs_list = probs.squeeze().cpu().tolist()
    pred_index = int(pred.item())
    class_names = artifact["class_names"]

    classes = [
        {
            "index": i,
            "label": class_names[i] if i < len(class_names) else f"class_{i}",
            "probability": probs_list[i]
        }
        for i in range(len(probs_list))
    ]

    result = {
        "predicted_index": pred_index,
        "predicted_label": class_names[pred_index] if pred_index < len(class_names) else f"class_{pred_index}",
        "classes": classes
    }

    print("inference reusult: ", result)
    return result

def run_segmentation(tif_path: Path, output_png: Path):
    print("Running segmentation task...")
    model, artifact = get_segmentation_model()

    bands = artifact["bands"]
    image_size = artifact["image_size"]

    with rasterio.open(tif_path) as src:
        full = src.read()  # (bands, H, W)

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

    H, W = x.shape[2], x.shape[3]

    # ---- NORMALIZATION (match training datamodule) ----
    means = torch.tensor(
        [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    ).view(1, -1, 1, 1)

    stds = torch.tensor(
        [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
    ).view(1, -1, 1, 1)

    x = (x - means) / stds

    x = F.interpolate(
        x,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).to(DEVICE)

    with torch.inference_mode():
        out = model(x)
        logits = out.output if hasattr(out, "output") else out
        pred = logits.argmax(dim=1).float()

    # resize mask back to original size
    pred = F.interpolate(
        pred.unsqueeze(1),
        size=(H, W),
        mode="nearest"
    ).squeeze()

    mask = pred.cpu().numpy().astype(np.uint8)

    # convert to colored PNG
    # class 1 = red overlay
    rgb = np.zeros((H, W, 4), dtype=np.uint8)
    rgb[..., 0] = mask * 255  # red
    rgb[..., 3] = mask * 255  # alpha

    Image.fromarray(rgb).save(output_png)

    print("Segmentation complete.")