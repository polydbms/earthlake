"""TerraMind model family benchmarker."""

from pathlib import Path
from typing import Dict, List, Any, Optional

from REMSA.models.base import BaseModelBenchmarker
from REMSA.datasets.registry import DatasetRegistry
from REMSA.utils.config_generator import TerraTorchConfigGenerator


class TerraMindBenchmarker(BaseModelBenchmarker):
    """Benchmarker for TerraMind foundation models."""

    VARIANTS = {
        "terramind_base_v1": {
            "backbone": "terramind_v1_base",
            "min_gpu_memory": 12,
            "description": "TerraMind v1 Base (~300M params multimodal EO model)",
            "pretrained": True,
        }
    }

    def get_model_family(self) -> str:
        return "terramind"

    def get_supported_variants(self) -> List[str]:
        return list(self.VARIANTS.keys())

    def get_model_config(self, variant: str) -> Dict[str, Any]:
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(
                f"Unknown TerraMind variant: {variant}. Available: {available}"
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

        if not self.validate_variant(variant):
            raise ValueError(f"Unsupported TerraMind variant: {variant}")

        model_config = self.get_model_config(variant)
        dataset_config = DatasetRegistry.get_dataset(dataset_name)

        # TerraMind requires modality + band mapping
        bands = [
            "BLUE",
            "GREEN",
            "RED",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2",
        ]

        extra = {
            "backbone_pretrained": True,
            "backbone_modalities": ["S2L1C"],
            "backbone_bands": {
                "S2L1C": bands
            },
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
            model_factory="EncoderDecoderFactory",
            band_requirements=bands,
            extra_model_args=extra,
            checkpoint_save_dir=checkpoint_save_dir,
        )

        return config_path