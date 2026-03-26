"""SatMAE model family benchmarker.

SatMAE requires a local clone of the SatMAE repository and pretrained
checkpoint files. Before using this benchmarker:

1. Clone the SatMAE repo:
       git clone https://github.com/sustainlab-group/SatMAE /path/to/SatMAE

2. Download pretrained checkpoints into a directory:
       pretrained-vit-base-e199.pth
       pretrained-vit-large-e199.pth

3. Set environment variables:
       export SATMAE_REPO_PATH=/path/to/SatMAE
       export SATMAE_CKPT_DIR=/path/to/checkpoints
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from REMSA.models.base import BaseModelBenchmarker
from REMSA.datasets.registry import DatasetRegistry
from REMSA.utils.config_generator import TerraTorchConfigGenerator


class SatMAEBenchmarker(BaseModelBenchmarker):
    """Benchmarker for SatMAE foundation models."""

    VARIANTS = {
        "satmae_vit_base": {
            "backbone": "vit_base_patch16",
            "model_factory": "SatMAEModelFactory",
            "min_gpu_memory": 8,
            "description": "SatMAE ViT-Base (86M params, fMoW RGB) [requires local repo]",
            "pretrained": True,
            "band_requirements": ["RED", "GREEN", "BLUE"],
            "checkpoint_file": "pretrained-vit-base-e199.pth",
        },
        "satmae_vit_large": {
            "backbone": "vit_large_patch16",
            "model_factory": "SatMAEModelFactory",
            "min_gpu_memory": 16,
            "description": "SatMAE ViT-Large (304M params, fMoW RGB/Sentinel-2) [requires local repo]",
            "pretrained": True,
            "band_requirements": ["RED", "GREEN", "BLUE"],
            "checkpoint_file": "pretrained-vit-large-e199.pth",
        },
    }

    def get_model_family(self) -> str:
        """Return model family name."""
        return "satmae"

    def get_supported_variants(self) -> List[str]:
        """Return list of supported SatMAE variants."""
        return list(self.VARIANTS.keys())

    def get_model_config(self, variant: str) -> Dict[str, Any]:
        """Return configuration for a specific SatMAE variant."""
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(
                f"Unknown SatMAE variant: {variant}. Available: {available}"
            )
        return self.VARIANTS[variant].copy()

    def _validate_satmae_environment(self, checkpoint_file: str) -> tuple:
        """Validate that SatMAE environment is properly configured.

        Lazily reads env vars and checks that paths exist on disk.

        Returns:
            Tuple of (repo_path, checkpoint_path) as strings.

        Raises:
            EnvironmentError: If env vars are missing or paths don't exist.
        """
        repo_path = os.environ.get("SATMAE_REPO_PATH", "")
        if not repo_path:
            raise EnvironmentError(
                "SATMAE_REPO_PATH environment variable is not set.\n\n"
                "SatMAE requires a local clone of the repository.\n"
                "Setup steps:\n"
                "  1. git clone https://github.com/sustainlab-group/SatMAE /path/to/SatMAE\n"
                "  2. export SATMAE_REPO_PATH=/path/to/SatMAE"
            )

        repo = Path(repo_path)
        if not repo.is_dir():
            raise EnvironmentError(
                f"SATMAE_REPO_PATH points to a non-existent directory: {repo_path}\n\n"
                "Please clone the SatMAE repository:\n"
                "  git clone https://github.com/sustainlab-group/SatMAE "
                f"{repo_path}"
            )

        ckpt_dir = os.environ.get("SATMAE_CKPT_DIR", "")
        if not ckpt_dir:
            raise EnvironmentError(
                "SATMAE_CKPT_DIR environment variable is not set.\n\n"
                "SatMAE requires pretrained checkpoint files.\n"
                "Setup steps:\n"
                "  1. Download checkpoints into a directory\n"
                "  2. export SATMAE_CKPT_DIR=/path/to/checkpoints"
            )

        ckpt_path = Path(ckpt_dir) / checkpoint_file
        if not ckpt_path.is_file():
            raise EnvironmentError(
                f"Checkpoint file not found: {ckpt_path}\n\n"
                f"Expected file '{checkpoint_file}' in SATMAE_CKPT_DIR={ckpt_dir}\n"
                "Download the pretrained checkpoints from the SatMAE repository."
            )

        return str(repo), str(ckpt_path)

    def create_config(
        self,
        variant: str,
        dataset_name: str,
        batch_size: int = 8,
        num_workers: int = 4,
        max_epochs: int = 10,
        learning_rate: float = 1e-4,
        mode: str = "test",
        checkpoint_save_dir: Optional[Path] = None,
    ) -> Path:
        """Create TerraTorch config file for SatMAE variant and dataset."""
        if not self.validate_variant(variant):
            raise ValueError(f"Unsupported SatMAE variant: {variant}")

        model_config = self.get_model_config(variant)

        # Validate environment before proceeding
        checkpoint_file = model_config.get("checkpoint_file", "")
        repo_path, checkpoint_path = self._validate_satmae_environment(
            checkpoint_file
        )

        dataset_config = DatasetRegistry.get_dataset(dataset_name)

        # SatMAE factory needs in_channels + bands + model_sys_path + checkpoint
        band_req = model_config.get("band_requirements")
        if band_req:
            extra = {
                "in_channels": len(band_req),
                "bands": band_req,
                "pretrained": True,
            }
        else:
            all_bands = dataset_config.get("all_hls_band_names", [])
            extra = {
                "in_channels": len(all_bands),
                "bands": all_bands,
                "pretrained": True,
            }

        extra["model_sys_path"] = repo_path
        extra["checkpoint_path"] = checkpoint_path

        generator = TerraTorchConfigGenerator()
        config_path = generator.create_config(
            model_family=self.get_model_family(),
            variant=variant,
            backbone=model_config["backbone"],
            dataset_config=dataset_config,
            output_dir=self.output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            mode=mode,
            model_factory=model_config.get("model_factory", "EncoderDecoderFactory"),
            band_requirements=band_req,
            extra_model_args=extra,
            checkpoint_save_dir=checkpoint_save_dir,
        )

        return config_path
