"""Registry for discovering models that support benchmarking."""

from pathlib import Path
from typing import Dict, List, Type, Optional, Any

from REMSA.models.base import BaseModelBenchmarker


class ModelRegistry:
    """Registry for model benchmarkers."""

    _benchmarkers: Dict[str, Type[BaseModelBenchmarker]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Lazily initialize with default benchmarkers."""
        if not cls._initialized:
            cls._register_defaults()
            cls._initialized = True

    @classmethod
    def _register_defaults(cls) -> None:
        """Register built-in benchmarkers."""
        from REMSA.models.prithvi import PrithviBenchmarker
        from REMSA.models.satmae import SatMAEBenchmarker
        from REMSA.models.scalemae import ScaleMAEBenchmarker
        from REMSA.models.satlas import SatlasBenchmarker
        from REMSA.models.ssl4eo import SSL4EOBenchmarker
        from REMSA.models.clay import ClayBenchmarker
        from REMSA.models.dofa import DOFABenchmarker
        from REMSA.models.terramind import TerraMindBenchmarker

        cls.register("prithvi", PrithviBenchmarker)
        cls.register("satmae", SatMAEBenchmarker)
        cls.register("scalemae", ScaleMAEBenchmarker)
        cls.register("satlas", SatlasBenchmarker)
        cls.register("ssl4eo", SSL4EOBenchmarker)
        cls.register("clay", ClayBenchmarker)
        cls.register("dofa", DOFABenchmarker)
        cls.register("terramind", TerraMindBenchmarker)

    @classmethod
    def register(
        cls, model_family: str, benchmarker_class: Type[BaseModelBenchmarker]
    ) -> None:
        """
        Register a benchmarker class for a model family.

        Args:
            model_family: Name of the model family (e.g., "prithvi")
            benchmarker_class: Benchmarker class to register
        """
        cls._benchmarkers[model_family] = benchmarker_class

    @classmethod
    def get_benchmarker(
        cls, model_family: str, output_dir: Path
    ) -> BaseModelBenchmarker:
        """
        Get a benchmarker instance for a model family.

        Args:
            model_family: Name of the model family
            output_dir: Directory for output files

        Returns:
            Benchmarker instance

        Raises:
            ValueError: If model family is not registered
        """
        cls._ensure_initialized()
        if model_family not in cls._benchmarkers:
            available = ", ".join(cls.list_families())
            raise ValueError(
                f"Unknown model family: {model_family}. Available: {available}"
            )
        return cls._benchmarkers[model_family](output_dir)

    @classmethod
    def list_families(cls) -> List[str]:
        """
        List all registered model families.

        Returns:
            List of model family names
        """
        cls._ensure_initialized()
        return list(cls._benchmarkers.keys())

    @classmethod
    def supports_benchmark(cls, model_name: str) -> bool:
        """
        Check if a model supports benchmarking.

        Args:
            model_name: Model family name or variant name

        Returns:
            True if model supports benchmarking
        """
        cls._ensure_initialized()

        # Direct family match
        if model_name in cls._benchmarkers:
            return True

        # Check if it's a variant of any registered family
        return cls.get_family_for_variant(model_name) is not None

    @classmethod
    def get_family_for_variant(cls, variant: str) -> Optional[str]:
        """
        Find which model family a variant belongs to.

        Args:
            variant: Variant name (e.g., "prithvi_100m")

        Returns:
            Model family name, or None if not found
        """
        cls._ensure_initialized()

        for family, benchmarker_cls in cls._benchmarkers.items():
            # Create temp instance to check variants
            temp = benchmarker_cls(Path("/tmp"))
            if variant in temp.get_supported_variants():
                return family

        return None

    @classmethod
    def list_all_variants(cls) -> Dict[str, List[str]]:
        """
        Get all model families with their variants.

        Returns:
            Dict mapping family names to lists of variant names
        """
        cls._ensure_initialized()

        result = {}
        for family, benchmarker_cls in cls._benchmarkers.items():
            temp = benchmarker_cls(Path("/tmp"))
            result[family] = temp.get_supported_variants()

        return result

    @classmethod
    def get_variant_info(cls, variant: str) -> Optional[Dict[str, Any]]:
        """
        Get info about a specific variant.

        Args:
            variant: Variant name (e.g., "prithvi_100m")

        Returns:
            Variant info dict, or None if not found
        """
        family = cls.get_family_for_variant(variant)
        if family is None:
            return None

        benchmarker = cls.get_benchmarker(family, Path("/tmp"))
        config = benchmarker.get_model_config(variant)
        config["family"] = family
        config["variant"] = variant

        return config
