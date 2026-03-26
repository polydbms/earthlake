"""DOFA model family benchmarker."""

from pathlib import Path
from typing import Dict, List, Any, Optional

from REMSA.models.base import BaseModelBenchmarker
from REMSA.datasets.registry import DatasetRegistry
from REMSA.utils.config_generator import TerraTorchConfigGenerator


class DOFABenchmarker(BaseModelBenchmarker):
    """Benchmarker for DOFA foundation models."""

    VARIANTS = {
        "dofa_base": {
            "backbone": "dofa_base_patch16_224",
            "min_gpu_memory": 8,
            "description": "DOFA Base (ViT-B, ~86M params, sensor-agnostic)",
            "pretrained": True,
        },
        "dofa_large": {
            "backbone": "dofa_large_patch16_224",
            "min_gpu_memory": 16,
            "description": "DOFA Large (ViT-L, ~300M params, sensor-agnostic)",
            "pretrained": True,
        },
    }

    def get_model_family(self) -> str:
        """Return model family name."""
        return "dofa"

    def get_supported_variants(self) -> List[str]:
        """Return list of supported DOFA variants."""
        return list(self.VARIANTS.keys())

    def get_model_config(self, variant: str) -> Dict[str, Any]:
        """Return configuration for a specific DOFA variant."""
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(
                f"Unknown DOFA variant: {variant}. Available: {available}"
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
        """Create TerraTorch config file for DOFA variant and dataset."""
        if not self.validate_variant(variant):
            raise ValueError(f"Unsupported DOFA variant: {variant}")

        model_config = self.get_model_config(variant)
        dataset_config = DatasetRegistry.get_dataset(dataset_name)

        # DOFA backbones need model_bands via backbone_kwargs
        band_req = model_config.get("band_requirements")
        if band_req:
            bands = band_req
        else:
            bands = dataset_config.get("all_hls_band_names", [])
        extra = {
            "backbone_kwargs": {"model_bands": bands, "img_size": 64},
            "necks": [
                {"name": "ReshapeTokensToImage", "remove_cls_token": True}
            ],
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
