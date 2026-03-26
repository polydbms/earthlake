#!/usr/bin/env python3
"""
TerraTorch Prithvi Benchmarking Script

This script runs Prithvi models on various benchmark datasets using
TerraTorch. Supports multiple Prithvi sizes (100M, 300M, 600M) and
multiple benchmark datasets.

Usage:
    python benchmark_prithvi.py --model prithvi_100m --dataset eurosat
    python benchmark_prithvi.py --model prithvi_300m \\
        --dataset bigearthnet --gpu 0
    python benchmark_prithvi.py --all --output results/
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional
import yaml


# Prithvi model configurations
PRITHVI_MODELS = {
    "prithvi_100m": {
        "backbone": "prithvi_eo_v1_100",
        "description": "Prithvi-EO-1.0 (100M parameters)",
        "min_gpu_memory": 8,  # GB
    },
    "prithvi_300m": {
        "backbone": "prithvi_eo_v2_300",
        "description": "Prithvi-EO-2.0 (300M parameters)",
        "min_gpu_memory": 16,  # GB
    },
    "prithvi_600m": {
        "backbone": "prithvi_eo_v2_600",
        "description": "Prithvi-EO-2.0 (600M parameters)",
        "min_gpu_memory": 24,  # GB
    },
}

# Benchmark dataset configurations
BENCHMARK_DATASETS = {
    "eurosat": {
        "task": "classification",
        "num_classes": 10,
        "size": "27k",
        "modality": "Sentinel-2 (13 bands)",
        "torchgeo": True,
        "data_module": "EuroSATDataModule",
    },
    "bigearthnet": {
        "task": "multi_label",
        "num_classes": 19,
        "size": "549k",
        "modality": "S1 + S2",
        "torchgeo": True,
        "data_module": "BigEarthNetDataModule",
    },
    "ucmerced": {
        "task": "classification",
        "num_classes": 21,
        "size": "2.1k",
        "modality": "RGB Aerial",
        "torchgeo": True,
        "data_module": "UCMercedDataModule",
    },
    "resisc45": {
        "task": "classification",
        "num_classes": 45,
        "size": "31.5k",
        "modality": "RGB",
        "torchgeo": True,
        "data_module": "RESISC45DataModule",
    },
    "hls_burn_scars": {
        "task": "segmentation",
        "num_classes": 2,
        "size": "~800",
        "modality": "HLS (6 bands)",
        "torchgeo": False,
        "huggingface": "ibm-nasa-geospatial/hls_burn_scars",
    },
    "sen1floods11": {
        "task": "segmentation",
        "num_classes": 2,
        "size": "4.8k",
        "modality": "S1 + S2",
        "torchgeo": True,
        "data_module": "Sen1Floods11DataModule",
    },
}


def create_config(
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    max_epochs: int = 10,
    learning_rate: float = 1e-4,
    image_size: int = 224,
) -> Path:
    """Create a TerraTorch YAML config file."""
    model_config = PRITHVI_MODELS[model_name]
    dataset_config = BENCHMARK_DATASETS[dataset_name]

    # Determine task class based on dataset task type
    if dataset_config["task"] == "classification":
        task_class = "terratorch.tasks.ClassificationTask"
    elif dataset_config["task"] == "multi_label":
        task_class = (
            "terratorch.tasks.MultiLabelClassificationTask"
        )
    elif dataset_config["task"] == "segmentation":
        task_class = "terratorch.tasks.SemanticSegmentationTask"
    else:
        raise ValueError(
            f"Unknown task type: {dataset_config['task']}"
        )

    # Build model args
    model_args = {
        "backbone": model_config["backbone"],
        "backbone_pretrained": True,
        "num_classes": dataset_config["num_classes"],
        # Tell model which HLS bands are in the input (order matters)
        "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    }

    # Add decoder based on task type
    if dataset_config["task"] == "segmentation":
        model_args["decoder"] = "UPerNetDecoder"
    else:
        # For classification tasks, use identity decoder
        model_args["decoder"] = "IdentityDecoder"

    # Build data module config
    if dataset_config.get("torchgeo"):
        dm_path = f"torchgeo.datamodules.{dataset_config['data_module']}"
        # TerraTorch uses dict_kwargs to pass params to underlying dataset
        dict_kwargs = {
            "root": str(output_dir / "data"),
            "download": True,
        }

        # Prithvi expects 6 HLS bands - select matching bands from EuroSAT
        # EuroSAT bands: B01-B12, B8A (13 total)
        # Prithvi needs: BLUE(B02), GREEN(B03), RED(B04), NIR_NARROW(B8A), SWIR_1(B11), SWIR_2(B12)
        if dataset_name == "eurosat":
            dict_kwargs["bands"] = ["B02", "B03", "B04", "B8A", "B11", "B12"]

        data_module = {
            "class_path": dm_path,
            "init_args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
            "dict_kwargs": dict_kwargs,
        }
    elif dataset_config.get("huggingface"):
        # For HuggingFace datasets, custom data module needed
        # Placeholder - adjust based on TerraTorch's HF support
        data_module = {
            "class_path": (
                "terratorch.datamodules.HuggingFaceDataModule"
            ),
            "init_args": {
                "dataset_name": dataset_config["huggingface"],
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
        }
    else:
        raise ValueError(
            f"No data module configuration for {dataset_name}"
        )

    # Determine loss function (terratorch only supports 'ce', 'bce', 'jaccard', 'focal')
    loss = (
        "bce"
        if dataset_config["task"] == "multi_label"
        else "ce"
    )

    # Build full config
    config = {
        "seed_everything": 42,
        "model": {
            "class_path": task_class,
            "init_args": {
                "model_factory": "EncoderDecoderFactory",
                "model_args": model_args,
                "loss": loss,
            },
        },
        "data": data_module,
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": max_epochs,
            "precision": "16-mixed",
            "log_every_n_steps": 10,
        },
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": learning_rate,
                "weight_decay": 0.01,
            },
        },
        "lr_scheduler": {
            "class_path": (
                "torch.optim.lr_scheduler.CosineAnnealingLR"
            ),
            "init_args": {
                "T_max": max_epochs,
            },
        },
    }

    # Save config
    config_path = (
        output_dir / "configs" / f"{model_name}_{dataset_name}.yaml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def run_benchmark(
    config_path: Path,
    output_dir: Path,
    mode: str = "test",
    checkpoint_path: Optional[Path] = None,
) -> Dict:
    """Run a TerraTorch benchmark."""
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = ["terratorch", mode, "--config", str(config_path)]

    if mode == "test" and checkpoint_path:
        cmd.extend(["--ckpt_path", str(checkpoint_path)])

    # Add output directory via trainer config
    cmd.extend(["--trainer.default_root_dir", str(results_dir)])

    print(f"Running: {' '.join(cmd)}")

    # Run command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "stdout": e.stdout,
            "stderr": e.stderr,
            "returncode": e.returncode,
        }


def check_gpu_memory(min_required: int) -> bool:
    """Check if GPU has sufficient memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            print(
                "Warning: CUDA not available. "
                "Running on CPU will be slow."
            )
            return False

        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
        )
        print(
            f"GPU Memory: {gpu_memory:.1f} GB "
            f"(Required: {min_required} GB)"
        )
        return gpu_memory >= min_required
    except ImportError:
        print(
            "Warning: PyTorch not installed. "
            "Cannot check GPU memory."
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Prithvi models using TerraTorch"
    )
    parser.add_argument(
        "--model",
        choices=list(PRITHVI_MODELS.keys()),
        help="Prithvi model to use",
    )
    parser.add_argument(
        "--dataset",
        choices=list(BENCHMARK_DATASETS.keys()),
        help="Benchmark dataset to use",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all model-dataset combinations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training/evaluation",
    )
    parser.add_argument(
        "--mode",
        choices=["fit", "test"],
        default="test",
        help="Mode: 'fit' for training, 'test' for evaluation only",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to checkpoint for test mode",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--skip_gpu_check",
        action="store_true",
        help="Skip GPU memory check",
    )

    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Determine which benchmarks to run
    if args.all:
        benchmarks = [
            (model, dataset)
            for model in PRITHVI_MODELS.keys()
            for dataset in BENCHMARK_DATASETS.keys()
        ]
    elif args.model and args.dataset:
        benchmarks = [(args.model, args.dataset)]
    else:
        parser.error(
            "Either --all or both --model and --dataset "
            "must be specified"
        )

    # Run benchmarks
    results = []
    for model_name, dataset_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {model_name} on {dataset_name}")
        print(f"{'='*60}")

        # Check GPU memory
        if not args.skip_gpu_check:
            min_memory = PRITHVI_MODELS[model_name]["min_gpu_memory"]
            if not check_gpu_memory(min_memory):
                print(
                    f"Warning: GPU memory may be insufficient "
                    f"for {model_name}"
                )
                response = input("Continue anyway? (y/n): ")
                if response.lower() != "y":
                    print("Skipping this benchmark...")
                    continue

        # Create config
        try:
            config_path = create_config(
                model_name=model_name,
                dataset_name=dataset_name,
                output_dir=args.output,
                batch_size=args.batch_size,
            )
            print(f"Created config: {config_path}")
        except Exception as e:
            print(f"Error creating config: {e}")
            results.append({
                "model": model_name,
                "dataset": dataset_name,
                "status": "config_error",
                "error": str(e),
            })
            continue

        # Run benchmark
        result = run_benchmark(
            config_path=config_path,
            output_dir=args.output,
            mode=args.mode,
            checkpoint_path=args.checkpoint,
        )

        results.append({
            "model": model_name,
            "dataset": dataset_name,
            "config": str(config_path),
            **result,
        })

        # Save intermediate results
        results_file = args.output / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Status: {result['status']}")

    # Print summary
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}")
    for result in results:
        status_icon = (
            "✅" if result["status"] == "success" else "❌"
        )
        print(
            f"{status_icon} {result['model']} on "
            f"{result['dataset']}: {result['status']}"
        )

    print(f"\nResults saved to: {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
