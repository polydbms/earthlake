"""SSL4EO-S12 model family benchmarker."""

from pathlib import Path
from typing import Dict, List, Any, Optional

from REMSA.models.base import BaseModelBenchmarker
from REMSA.datasets.registry import DatasetRegistry
from REMSA.utils.config_generator import TerraTorchConfigGenerator


class SSL4EOBenchmarker(BaseModelBenchmarker):
    """Benchmarker for SSL4EO-S12 foundation models."""

    VARIANTS = {
        "ssl4eo_resnet50_s2_all_moco": {
            "backbone": "ssl4eos12_resnet50_sentinel2_all_moco",
            "min_gpu_memory": 4,
            "description": "SSL4EO-S12 ResNet-50, Sentinel-2 all 13 bands, MoCo (25M params)",
            "pretrained": True,
        },
        "ssl4eo_resnet50_s2_all_dino": {
            "backbone": "ssl4eos12_resnet50_sentinel2_all_dino",
            "min_gpu_memory": 4,
            "description": "SSL4EO-S12 ResNet-50, Sentinel-2 all 13 bands, DINO (25M params)",
            "pretrained": True,
        },
        "ssl4eo_resnet50_s2_rgb_moco": {
            "backbone": "ssl4eos12_resnet50_sentinel2_rgb_moco",
            "min_gpu_memory": 4,
            "description": "SSL4EO-S12 ResNet-50, Sentinel-2 RGB, MoCo (25M params)",
            "pretrained": True,
            "band_requirements": ["RED", "GREEN", "BLUE"],
        },
        "ssl4eo_resnet50_s1_all_moco": {
            "backbone": "ssl4eos12_resnet50_sentinel1_all_moco",
            "min_gpu_memory": 4,
            "description": "SSL4EO-S12 ResNet-50, Sentinel-1 SAR, MoCo (25M params)",
            "pretrained": True,
        },
        "ssl4eo_vit_small_s2_all_dino": {
            "backbone": "ssl4eos12_vit_small_patch16_224_sentinel2_all_dino",
            "min_gpu_memory": 4,
            "description": "SSL4EO-S12 ViT-Small, Sentinel-2 all bands, DINO (22M params)",
            "pretrained": True,
        },
    }

    def get_model_family(self) -> str:
        """Return model family name."""
        return "ssl4eo"

    def get_supported_variants(self) -> List[str]:
        """Return list of supported SSL4EO-S12 variants."""
        return list(self.VARIANTS.keys())

    def get_model_config(self, variant: str) -> Dict[str, Any]:
        """Return configuration for a specific SSL4EO-S12 variant."""
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(
                f"Unknown SSL4EO variant: {variant}. Available: {available}"
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
        """Create TerraTorch config file for SSL4EO-S12 variant and dataset."""
        if not self.validate_variant(variant):
            raise ValueError(f"Unsupported SSL4EO variant: {variant}")

        model_config = self.get_model_config(variant)
        dataset_config = DatasetRegistry.get_dataset(dataset_name)

        # SSL4EO backbones need model_bands via backbone_kwargs
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
