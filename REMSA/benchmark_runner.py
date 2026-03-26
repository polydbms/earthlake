"""Main API for the benchmarking system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from REMSA.config import config
from REMSA.datasets.registry import DatasetRegistry
from REMSA.models.registry import ModelRegistry
from REMSA.utils.job_runner import JobRunner, BenchmarkJob


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    model_variant: str
    dataset: str
    status: str  # "success", "failed", "error"
    metrics: Optional[Dict[str, float]] = None
    config_path: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None


class BenchmarkRunner:
    """
    Main API for the benchmarking system.

    Designed for UI integration - provides methods to check benchmark
    availability, run benchmarks, and retrieve results.

    Example:
        runner = BenchmarkRunner()

        # Check if button should appear
        if runner.supports_benchmark("prithvi_100m"):
            # Run benchmark when clicked
            result = runner.run_benchmark("prithvi_100m", "eurosat")
            print(result.metrics)
    """

    def __init__(self, output_dir: Path = Path("benchmark_results")):
        """
        Initialize BenchmarkRunner.

        Args:
            output_dir: Directory for benchmark outputs (configs, results)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        weights_dir = config.get("MODEL_WEIGHTS_DIR", "model_weights")
        self.model_weights_dir = Path(weights_dir)
        self.finetuned_dir = self.model_weights_dir / "finetuned"
        self.finetuned_dir.mkdir(parents=True, exist_ok=True)

        executor = config.get("BENCHMARK_EXECUTOR", "local")
        if executor == "slurm":
            from REMSA.utils.slurm_runner import SlurmJobRunner
            self.job_runner = SlurmJobRunner(
                output_dir=self.output_dir,
                host=config["SLURM_HOST"],
                user=config["SLURM_USER"],
                password=config["SLURM_PASSWORD"],
                work_dir=config.get("SLURM_WORK_DIR", "/tmp/benchmarks"),
                partition=config.get("SLURM_PARTITION", "gpu"),
                account=config.get("SLURM_ACCOUNT"),
                conda_env=config.get("SLURM_CONDA_ENV"),
                conda_base=config.get("SLURM_CONDA_BASE"),
                modules=config.get("SLURM_MODULES"),
            )
        else:
            self.job_runner = JobRunner(self.output_dir)

        self._executor = executor

    # === Checkpoint helpers ===

    def _stable_checkpoint_dir(self, model_variant: str, dataset: str) -> Path:
        return self.finetuned_dir / model_variant / dataset

    def _stable_checkpoint_path(self, model_variant: str, dataset: str) -> Path:
        return self._stable_checkpoint_dir(model_variant, dataset) / "checkpoint-last.ckpt"

    # === UI Query Methods ===

    def supports_benchmark(self, model_name: str) -> bool:
        """
        Check if a model supports benchmarking.

        Use this to determine if "Run Benchmark" button should appear.

        Args:
            model_name: Model variant or family name

        Returns:
            True if model supports benchmarking
        """
        return ModelRegistry.supports_benchmark(model_name)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get all models that support benchmarking.

        Returns:
            List of model info dicts with keys:
                - family: Model family name
                - variant: Variant name
                - description: Human-readable description
                - min_gpu_memory: Required GPU memory in GB
        """
        result = []

        for family, variants in ModelRegistry.list_all_variants().items():
            benchmarker = ModelRegistry.get_benchmarker(family, self.output_dir)

            for variant in variants:
                config = benchmarker.get_model_config(variant)
                result.append({
                    "family": family,
                    "variant": variant,
                    "description": config.get("description", ""),
                    "min_gpu_memory": config.get("min_gpu_memory", 8),
                })

        return result

    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Get all available benchmark datasets.

        Returns:
            List of dataset info dicts
        """
        return [
            DatasetRegistry.get_dataset(name)
            for name in DatasetRegistry.list_datasets()
        ]

    def get_datasets_by_task(self, task_type: str) -> List[str]:
        """
        Get datasets for a specific task type.

        Args:
            task_type: "classification", "multi_label", or "segmentation"

        Returns:
            List of dataset names
        """
        return DatasetRegistry.get_datasets_by_task(task_type)

    # === Benchmark Execution ===

    def run_benchmark(
        self,
        model_variant: str,
        dataset: str,
        job_id: str,
        mode: str = "test",
        batch_size: int = 8,
        max_epochs: int = 10,
        learning_rate: float = 1e-4,
        num_workers: int = 4,
        checkpoint_path: Optional[Path] = None,
        on_process_started: Optional[callable] = None,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            model_variant: Model variant name (e.g., "prithvi_100m")
            dataset: Dataset name (e.g., "eurosat")
            mode: "fit" for training, "test" for evaluation
            batch_size: Batch size
            max_epochs: Maximum epochs (for fit mode)
            learning_rate: Learning rate (for fit mode)
            num_workers: Data loader workers
            checkpoint_path: Checkpoint for test mode

        Returns:
            BenchmarkResult with status and metrics
        """
        # Validate dataset
        if not DatasetRegistry.validate_dataset(dataset):
            return BenchmarkResult(
                model_variant=model_variant,
                dataset=dataset,
                status="error",
                error_message=f"Unknown dataset: {dataset}",
            )

        # Find model family
        family = ModelRegistry.get_family_for_variant(model_variant)
        if family is None:
            return BenchmarkResult(
                model_variant=model_variant,
                dataset=dataset,
                status="error",
                error_message=f"Unknown model variant: {model_variant}",
            )

        # Checkpoint reuse: if a fine-tuned checkpoint exists, skip training
        stable_ckpt = self._stable_checkpoint_path(model_variant, dataset)
        cached = stable_ckpt.exists()

        if cached:
            # Reuse fine-tuned checkpoint for both local and SLURM
            effective_mode = "test"
            checkpoint_path = stable_ckpt
            ckpt_save_dir = None
            ckpt_download_path = None
        else:
            effective_mode = mode
            if mode in ["linear_probe", "finetune"]:
                if self._executor == "slurm":
                    ckpt_save_dir = None                # no ModelCheckpoint injection (runs remotely)
                    ckpt_download_path = stable_ckpt    # download after fit completes
                else:
                    ckpt_save_dir = self._stable_checkpoint_dir(model_variant, dataset)
                    ckpt_download_path = None           # saved locally via ModelCheckpoint callback
            else:
                ckpt_save_dir = None
                ckpt_download_path = None

        # Create config
        try:
            benchmarker = ModelRegistry.get_benchmarker(family, self.output_dir)
            config_path = benchmarker.create_config(
                variant=model_variant,
                dataset_name=dataset,
                batch_size=batch_size,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                num_workers=num_workers,
                mode=effective_mode,
                checkpoint_save_dir=ckpt_save_dir,
            )
        except Exception as e:
            return BenchmarkResult(
                model_variant=model_variant,
                dataset=dataset,
                status="error",
                error_message=f"Config creation failed: {e}",
            )

        # Create and run job
        job = BenchmarkJob(
            model_variant=model_variant,
            dataset=dataset,
            config_path=config_path,
            job_id=job_id
        )

        if effective_mode in ["linear_probe", "finetune"]:
            cli_mode = "fit"
        else:
            cli_mode = "test"

        job = self.job_runner.run_job(
            job,
            mode=cli_mode,
            checkpoint_path=checkpoint_path,
            checkpoint_save_path=ckpt_download_path,
            on_process_started=on_process_started,
        )

        # Convert to result
        return BenchmarkResult(
            model_variant=model_variant,
            dataset=dataset,
            status="success" if job.status == "completed" else "failed",
            metrics=job.result.get("metrics") if job.result else None,
            config_path=str(stable_ckpt) if cached else str(config_path),
            error_message=job.error,
            duration_seconds=job.duration_seconds,
        )

    def run_benchmark_suite(
        self,
        model_variant: str,
        datasets: Optional[List[str]] = None,
        **kwargs,
    ) -> List[BenchmarkResult]:
        """
        Run a model on multiple datasets.

        Args:
            model_variant: Model variant name
            datasets: List of dataset names (None = all datasets)
            **kwargs: Additional arguments passed to run_benchmark

        Returns:
            List of BenchmarkResult objects
        """
        if datasets is None:
            datasets = DatasetRegistry.list_datasets()

        results = []
        for dataset in datasets:
            result = self.run_benchmark(model_variant, dataset, **kwargs)
            results.append(result)

        return results

    def check_gpu_requirements(self, model_variant: str) -> Dict[str, Any]:
        """
        Check if GPU meets requirements for a model variant.

        Args:
            model_variant: Model variant name

        Returns:
            Dict with:
                - meets_requirements: bool
                - required_memory: int (GB)
                - available_memory: float (GB) or None if no GPU
        """
        info = ModelRegistry.get_variant_info(model_variant)
        if info is None:
            return {
                "meets_requirements": False,
                "required_memory": 0,
                "available_memory": None,
                "error": f"Unknown variant: {model_variant}",
            }

        required = info.get("min_gpu_memory", 8)

        # In SLURM mode, GPU is available on the remote cluster
        if self._executor == "slurm":
            return {
                "meets_requirements": True,
                "required_memory": required,
                "available_memory": None,
                "note": "Will run on remote SLURM cluster",
            }

        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    "meets_requirements": False,
                    "required_memory": required,
                    "available_memory": None,
                    "error": "No GPU available",
                }

            available = (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
            )
            return {
                "meets_requirements": available >= required,
                "required_memory": required,
                "available_memory": round(available, 1),
            }

        except ImportError:
            return {
                "meets_requirements": False,
                "required_memory": required,
                "available_memory": None,
                "error": "PyTorch not installed",
            }
