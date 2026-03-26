"""Satlas model family benchmarker."""

from pathlib import Path
from typing import Dict, List, Any, Optional

from REMSA.models.base import BaseModelBenchmarker
from REMSA.datasets.registry import DatasetRegistry
from REMSA.utils.config_generator import TerraTorchConfigGenerator


class SatlasBenchmarker(BaseModelBenchmarker):
    """Benchmarker for Satlas foundation models."""

    VARIANTS = {
        "satlas_swin_b_s2_si_ms": {
            "backbone": "satlas_swin_b_sentinel2_si_ms",
            "min_gpu_memory": 8,
            "description": "Satlas Swin-v2 Base, Sentinel-2 multispectral single-image (~100M params)",
            "pretrained": True,
        },
        "satlas_swin_b_s2_si_rgb": {
            "backbone": "satlas_swin_b_sentinel2_si_rgb",
            "min_gpu_memory": 8,
            "description": "Satlas Swin-v2 Base, Sentinel-2 RGB single-image (~100M params)",
            "pretrained": True,
        },
        "satlas_swin_b_s2_mi_ms": {
            "backbone": "satlas_swin_b_sentinel2_mi_ms",
            "min_gpu_memory": 12,
            "description": "Satlas Swin-v2 Base, Sentinel-2 multispectral multi-image (~100M params)",
            "pretrained": True,
        },
        "satlas_resnet50_s2_si_ms": {
            "backbone": "satlas_resnet50_sentinel2_si_ms_satlas",
            "min_gpu_memory": 4,
            "description": "Satlas ResNet-50, Sentinel-2 multispectral single-image (25M params)",
            "pretrained": True,
        },
    }

    def get_model_family(self) -> str:
        """Return model family name."""
        return "satlas"

    def get_supported_variants(self) -> List[str]:
        """Return list of supported Satlas variants."""
        return list(self.VARIANTS.keys())

    def get_model_config(self, variant: str) -> Dict[str, Any]:
        """Return configuration for a specific Satlas variant."""
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(
                f"Unknown Satlas variant: {variant}. Available: {available}"
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
        """Create TerraTorch config file for Satlas variant and dataset."""
        if not self.validate_variant(variant):
            raise ValueError(f"Unsupported Satlas variant: {variant}")

        model_config = self.get_model_config(variant)
        dataset_config = DatasetRegistry.get_dataset(dataset_name)

        # Satlas backbones need model_bands via backbone_kwargs
        band_req = model_config.get("band_requirements")
        if band_req:
            bands = band_req
        else:
            bands = dataset_config.get("all_hls_band_names", [])
        extra = {"backbone_kwargs": {"model_bands": bands}}

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
