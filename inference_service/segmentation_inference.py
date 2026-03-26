import torch
import rasterio
import numpy as np
import torch.nn.functional as F
import yaml
from pathlib import Path
from functools import lru_cache
from terratorch.tasks import SemanticSegmentationTask

from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=3)  # Cache one of each flavor
def load_model(job_id: str):
    # Adjust path to your results directory
    job_dir = Path(f"/app/benchmark_results/results/{job_id}")

    # Find config - flexible for different versioning
    deploy_cfg = next(job_dir.rglob("config_deploy.yaml"))
    with open(deploy_cfg) as f:
        cfg = yaml.safe_load(f)

    # Load latest checkpoint
    ckpt = sorted(
        [p for p in job_dir.rglob("*.ckpt") if not p.name.endswith("_state_dict.ckpt")],
        key=lambda p: p.stat().st_mtime,
    )[-1]

    model = SemanticSegmentationTask.load_from_checkpoint(str(ckpt))
    model.eval()
    model.to(DEVICE)

    init_args = cfg["data"]["init_args"]
    model_args = cfg["model"]["init_args"]["model_args"]

    artifact = {
        "bands": init_args.get("output_bands"),
        "means": init_args.get("means"),
        "stds": init_args.get("stds"),
        "backbone": model_args.get("backbone", ""),
        # Capture modalities for TerraMind
        "modalities": model_args.get("backbone_modalities", None)
    }

    return model, artifact


def run(job_id: str, tif_path: Path, output_png: Path):
    model, artifact = load_model(job_id)

    with rasterio.open(tif_path) as src:
        full = src.read()
        H_orig, W_orig = full.shape[1], full.shape[2]  # 512, 512

    # 1. Band Mapping & Normalization
    bands = artifact["bands"]
    S2_INDEX = {
        "COASTAL_AEROSOL": 0, "BLUE": 1, "GREEN": 2, "RED": 3,
        "RED_EDGE_1": 4, "RED_EDGE_2": 5, "RED_EDGE_3": 6, "NIR_BROAD": 7,
        "NIR_NARROW": 8, "WATER_VAPOR": 9, "CIRRUS": 10, "SWIR_1": 11, "SWIR_2": 12,
    }
    x_np = np.stack([full[S2_INDEX[b]] for b in bands], axis=0)
    x = torch.from_numpy(x_np).float().unsqueeze(0).to(DEVICE)

    means = torch.tensor(artifact["means"], device=DEVICE).view(1, -1, 1, 1)
    stds = torch.tensor(artifact["stds"], device=DEVICE).view(1, -1, 1, 1)
    x = (x - means) / stds

    # 2. Manual Tiling Logic (Handle 512x512 -> 256x256)
    tile_size = 256
    full_mask = torch.zeros((H_orig, W_orig), device=DEVICE)

    with torch.inference_mode():
        # Loop through the image in 256-pixel increments
        for i in range(0, H_orig, tile_size):
            for j in range(0, W_orig, tile_size):
                # Extract the tile
                tile = x[:, :, i:i + tile_size, j:j + tile_size]

                # Check if we need TerraMind modalities
                if "terramind" in artifact["backbone"].lower():
                    logits = model(tile, modalities=artifact["modalities"])
                else:
                    logits = model(tile)

                logits = logits.output if hasattr(logits, "output") else logits
                tile_pred = logits.argmax(dim=1).squeeze()  # Shape [256, 256]

                # Place the tile prediction back into the full mask
                full_mask[i:i + tile_size, j:j + tile_size] = tile_pred

    # 3. Final RGBA Generation
    mask = full_mask.cpu().numpy().astype(np.uint8)
    rgba = np.zeros((H_orig, W_orig, 4), dtype=np.uint8)
    rgba[..., 0] = mask * 255
    rgba[..., 3] = mask * 255

    Image.fromarray(rgba).save(output_png)