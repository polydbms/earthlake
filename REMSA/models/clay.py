"""Clay model family benchmarker."""

from pathlib import Path
from typing import Dict, List, Any, Optional

from REMSA.models.base import BaseModelBenchmarker
from REMSA.datasets.registry import DatasetRegistry
from REMSA.utils.config_generator import TerraTorchConfigGenerator

# Sentinel-2 L2A metadata for Clay v1.5
# Wavelengths in micrometers, matching EuroSAT's 13-band order
SENTINEL2_METADATA = {
    "sentinel-2-l2a": {
        "gsd": 10,
        "rgb_indices": [3, 2, 1],  # RED, GREEN, BLUE indices in band order
        "bands": {
            "wavelength": {
                "B01": 0.443,
                "B02": 0.490,
                "B03": 0.560,
                "B04": 0.665,
                "B05": 0.704,
                "B06": 0.740,
                "B07": 0.783,
                "B08": 0.842,
                "B09": 0.945,
                "B10": 1.375,
                "B11": 1.610,
                "B12": 2.190,
                "B8A": 0.865,
            },
        },
    },
}

# Clay v1.5 architecture hyperparameters per variant
CLAY_ARCH_PARAMS = {
    "clay_v15_small": {
        "dim": 384,
        "depth": 6,
        "heads": 6,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 192,
        "decoder_depth": 2,
        "decoder_heads": 6,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    },
    "clay_v15_base": {
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 384,
        "decoder_depth": 4,
        "decoder_heads": 12,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    },
    "clay_v15_large": {
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 6,
        "decoder_heads": 16,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    },
}


class ClayBenchmarker(BaseModelBenchmarker):
    """Benchmarker for Clay foundation models."""

    VARIANTS = {
        "clay_v1_base": {
            "backbone": "clay_v1_base",
            "model_factory": "EncoderDecoderFactory",
            "min_gpu_memory": 8,
            "description": "Clay v1 Base (~100M params, MAE+DINO, multi-sensor)",
            "pretrained": True,
        },
        "clay_v15_small": {
            "backbone": "clay_mae_small",
            "model_factory": "Clay1_5ModelFactory",
            "min_gpu_memory": 4,
            "description": "Clay v1.5 Small (~30M params)",
            "pretrained": True,
        },
        "clay_v15_base": {
            "backbone": "clay_mae_base",
            "model_factory": "Clay1_5ModelFactory",
            "min_gpu_memory": 8,
            "description": "Clay v1.5 Base (~100M params)",
            "pretrained": True,
        },
        "clay_v15_large": {
            "backbone": "clay_mae_large",
            "model_factory": "Clay1_5ModelFactory",
            "min_gpu_memory": 16,
            "description": "Clay v1.5 Large (~311M params)",
            "pretrained": True,
        },
    }

    def get_model_family(self) -> str:
        """Return model family name."""
        return "clay"

    def get_supported_variants(self) -> List[str]:
        """Return list of supported Clay variants."""
        return list(self.VARIANTS.keys())

    def get_model_config(self, variant: str) -> Dict[str, Any]:
        """Return configuration for a specific Clay variant."""
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(
                f"Unknown Clay variant: {variant}. Available: {available}"
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
        """Create TerraTorch config file for Clay variant and dataset."""
        if not self.validate_variant(variant):
            raise ValueError(f"Unsupported Clay variant: {variant}")

        model_config = self.get_model_config(variant)
        dataset_config = DatasetRegistry.get_dataset(dataset_name)

        bands = [
            "BLUE",
            "GREEN",
            "RED",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2",
        ]

        # Clay factory needs in_channels
        required_bands = bands

        extra = {
            "backbone_pretrained": True,
            "backbone_bands": bands,
        }

        #extra = {
        #    "in_channels": len(required_bands) if required_bands else 13,
        #}

        # Clay1_5ModelFactory passes all **kwargs to ClayMAE constructor.
        # It needs: batch_size, platform, metadata, and all architecture params.
        #factory = model_config.get("model_factory", "")
        #if "Clay1_5" in factory:
        #    extra["batch_size"] = batch_size
        #    extra["platform"] = ["sentinel-2-l2a"]
        #    extra["metadata"] = SENTINEL2_METADATA

            # ClayMAE training/inference params
        #    extra["mask_ratio"] = 0.0  # no masking for fine-tuning
        #    extra["patch_size"] = 8
        #    extra["norm_pix_loss"] = False
        #    extra["shuffle"] = False
        #    extra["teacher"] = "vit_base_patch14_reg4_dinov2.lvd142m"
        #    extra["dolls"] = []
        #    extra["doll_weights"] = []

            # Architecture params for this variant
        #    arch = CLAY_ARCH_PARAMS.get(variant, {})
        #    extra.update(arch)

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
            band_requirements=required_bands,
            extra_model_args=extra,
            checkpoint_save_dir=checkpoint_save_dir,
        )

        return config_path
