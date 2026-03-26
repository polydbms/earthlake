"""Abstract base class for all model benchmarkers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
from REMSA.datasets.registry import DatasetRegistry


class BaseModelBenchmarker(ABC):
    """Abstract base class for all model benchmarkers."""

    def __init__(self, output_dir: Path):
        """
        Initialize benchmarker.

        Args:
            output_dir: Directory for output files (configs, results)
        """
        self.output_dir = output_dir
        self.dataset_registry = DatasetRegistry()

    @abstractmethod
    def get_model_family(self) -> str:
        """
        Return model family name.

        Returns:
            Model family name (e.g., 'prithvi', 'satmae')
        """
        pass

    @abstractmethod
    def get_supported_variants(self) -> List[str]:
        """
        Return list of supported model variants.

        Returns:
            List of variant names (e.g., ['prithvi_100m', 'prithvi_300m'])
        """
        pass

    @abstractmethod
    def get_model_config(self, variant: str) -> Dict[str, Any]:
        """
        Return model-specific configuration for a variant.

        Args:
            variant: Model variant name (e.g., 'prithvi_100m')

        Returns:
            Dictionary with:
                - "backbone": str - TerraTorch backbone name
                - "min_gpu_memory": int - GB required
                - "description": str - Human-readable description
                - "pretrained": bool - Whether pretrained weights available

        Raises:
            ValueError: If variant is not supported
        """
        pass

    @abstractmethod
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
        """
        Create TerraTorch config file for this model variant and dataset.

        Args:
            variant: Model variant name
            dataset_name: Dataset name from registry
            batch_size: Batch size for training/evaluation
            num_workers: Number of data loader workers
            max_epochs: Maximum training epochs
            learning_rate: Learning rate
            mode: "fit" for training, "test" for evaluation only

        Returns:
            Path to created YAML config file
        """
        pass

    def validate_variant(self, variant: str) -> bool:
        """
        Check if variant is supported.

        Args:
            variant: Model variant name to validate

        Returns:
            True if variant is supported, False otherwise
        """
        return variant in self.get_supported_variants()

    def get_required_gpu_memory(self, variant: str) -> int:
        """
        Get minimum GPU memory required for variant.

        Args:
            variant: Model variant name

        Returns:
            Minimum GPU memory in GB
        """
        config = self.get_model_config(variant)
        return config.get("min_gpu_memory", 8)
