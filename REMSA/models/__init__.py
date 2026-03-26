"""Model benchmarkers package."""

from REMSA.models.base import BaseModelBenchmarker
from REMSA.models.registry import ModelRegistry
from REMSA.models.prithvi import PrithviBenchmarker
from REMSA.models.satmae import SatMAEBenchmarker
from REMSA.models.scalemae import ScaleMAEBenchmarker
from REMSA.models.satlas import SatlasBenchmarker
from REMSA.models.ssl4eo import SSL4EOBenchmarker
from REMSA.models.clay import ClayBenchmarker
from REMSA.models.dofa import DOFABenchmarker

__all__ = [
    "BaseModelBenchmarker",
    "ModelRegistry",
    "PrithviBenchmarker",
    "SatMAEBenchmarker",
    "ScaleMAEBenchmarker",
    "SatlasBenchmarker",
    "SSL4EOBenchmarker",
    "ClayBenchmarker",
    "DOFABenchmarker",
]
