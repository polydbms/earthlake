"""Prithvi model family benchmarker."""

from pathlib import Path
from typing import Dict, List, Any, Optional

from REMSA.models.base import BaseModelBenchmarker
from REMSA.datasets.registry import DatasetRegistry
from REMSA.utils.config_generator import TerraTorchConfigGenerator


class PrithviBenchmarker(BaseModelBenchmarker):
    """Benchmarker for Prithvi foundation models."""

    VARIANTS = {
        "prithvi_100m": {
            "backbone": "prithvi_eo_v1_100",
            "min_gpu_memory": 8,
            "description": "Prithvi-EO-1.0 (100M parameters)",
            "pretrained": True,
        },
        "prithvi_300m": {
            "backbone": "prithvi_eo_v2_300",
            "min_gpu_memory": 16,
            "description": "Prithvi-EO-2.0 (300M parameters)",
            "pretrained": True,
        },
        "prithvi_600m": {
            "backbone": "prithvi_eo_v2_600",
            "min_gpu_memory": 24,
            "description": "Prithvi-EO-2.0 (600M parameters)",
            "pretrained": True,
        },
    }

    def get_model_family(self) -> str:
        """Return model family name."""
        return "prithvi"

    def get_supported_variants(self) -> List[str]:
        """Return list of supported Prithvi variants."""
        return list(self.VARIANTS.keys())

    def get_model_config(self, variant: str) -> Dict[str, Any]:
        """Return configuration for a specific Prithvi variant."""
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(
                f"Unknown Prithvi variant: {variant}. Available: {available}"
            )
        return self.VARIANTS[variant].copy()

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
        """Create TerraTorch config file for Prithvi variant and dataset."""
        if not self.validate_variant(variant):
            raise ValueError(f"Unsupported Prithvi variant: {variant}")

        model_config = self.get_model_config(variant)
        dataset_config = DatasetRegistry.get_dataset(dataset_name)

        # Prithvi uses backbone_bands (HLS band names)
        band_req = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
        extra = {
            "backbone_pretrained": True,
            "backbone_bands": band_req,
        }

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
            band_requirements=band_req,
            extra_model_args=extra,
            checkpoint_save_dir=checkpoint_save_dir,
        )

        return config_path
