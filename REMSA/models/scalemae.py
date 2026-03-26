"""ScaleMAE model family benchmarker."""

from pathlib import Path
from typing import Dict, List, Any, Optional

from REMSA.models.base import BaseModelBenchmarker
from REMSA.datasets.registry import DatasetRegistry
from REMSA.utils.config_generator import TerraTorchConfigGenerator


class ScaleMAEBenchmarker(BaseModelBenchmarker):
    """Benchmarker for ScaleMAE foundation models."""

    VARIANTS = {
        "scalemae_large": {
            "backbone": "scalemae_large_patch16",
            "min_gpu_memory": 16,
            "description": "ScaleMAE ViT-Large (304M params, fMoW RGB)",
            "pretrained": True,
            "band_requirements": ["RED", "GREEN", "BLUE"],
        },
    }

    def get_model_family(self) -> str:
        """Return model family name."""
        return "scalemae"

    def get_supported_variants(self) -> List[str]:
        """Return list of supported ScaleMAE variants."""
        return list(self.VARIANTS.keys())

    def get_model_config(self, variant: str) -> Dict[str, Any]:
        """Return configuration for a specific ScaleMAE variant."""
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(
                f"Unknown ScaleMAE variant: {variant}. Available: {available}"
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
        """Create TerraTorch config file for ScaleMAE variant and dataset."""
        if not self.validate_variant(variant):
            raise ValueError(f"Unsupported ScaleMAE variant: {variant}")

        model_config = self.get_model_config(variant)
        dataset_config = DatasetRegistry.get_dataset(dataset_name)

        # ScaleMAE is natively supported in TorchGeo; use backbone_pretrained
        # and backbone_bands consistent with EncoderDecoderFactory (like Prithvi)
        band_req = model_config.get(
            "band_requirements", ["RED", "GREEN", "BLUE"]
        )
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
