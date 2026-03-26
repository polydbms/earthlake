"""Job execution utilities for running TerraTorch benchmarks."""

import re
import subprocess
import uuid
import sys
import torch
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

use_gpu = torch.cuda.is_available()

@dataclass
class BenchmarkJob:
    """Represents a benchmark job."""
    job_id: str
    model_variant: str = ""
    dataset: str = ""
    config_path: Optional[Path] = None
    #job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class JobRunner:
    """Executes benchmark jobs using TerraTorch CLI."""

    def __init__(self, output_dir: Path):
        """
        Initialize JobRunner.

        Args:
            output_dir: Directory for job outputs
        """
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_job(
        self,
        job: BenchmarkJob,
        mode: str = "test",
        checkpoint_path: Optional[Path] = None,
        checkpoint_save_path: Optional[Path] = None,
        on_process_started: Optional[callable] = None,
    ) -> BenchmarkJob:
        """
        Execute a benchmark job synchronously.

        Args:
            job: BenchmarkJob to execute
            mode: TerraTorch mode ("fit" or "test")
            checkpoint_path: Optional checkpoint for test mode

        Returns:
            Updated BenchmarkJob with results
        """
        job.status = "running"
        job.started_at = datetime.now()

        # Build command
        cmd = [
            "terratorch",
            mode,
            "--config",
            str(job.config_path),
        ]

        if mode == "test" and checkpoint_path:
            cmd.extend(["--ckpt_path", str(checkpoint_path)])

        if use_gpu:
            cmd.append("--trainer.accelerator=gpu")
            cmd.append("--trainer.precision=16-mixed")
        else:
            cmd.append("--trainer.accelerator=cpu")
            cmd.append("--trainer.precision=32-true")

        # Add output directory for this job
        job_output_dir = self.results_dir / job.job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        #cmd.extend(["--output_dir", str(job_output_dir)])
        cmd.append(f"--trainer.default_root_dir={job_output_dir}")
        #cmd.append(f"--trainer.logger.init_args.save_dir={job_output_dir}")
        cmd.append("--trainer.logger=lightning.pytorch.loggers.CSVLogger")
        cmd.append(f"--trainer.logger.init_args.save_dir={job_output_dir}")
        cmd.append("--trainer.logger.init_args.name=lightning_logs")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # expose process so API can cancel it
            job.process = process

            if on_process_started:
                on_process_started(process)

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

            job.status = "completed"
            job.result = {
                "stdout": stdout,
                "stderr": stderr,
                "metrics": self._parse_metrics_from_csv(job_output_dir),
                "output_dir": str(job_output_dir),
            }

        except subprocess.CalledProcessError as e:
            job.status = "failed"
            job.error = (
                f"TerraTorch failed (exit {e.returncode})\n\n"
                f"STDOUT:\n{e.stdout}\n\n"
                f"STDERR:\n{e.stderr}"
            )
            job.result = {
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
                "metrics": {},
            }

        except FileNotFoundError:
            job.status = "failed"
            job.error = "TerraTorch not found. Is it installed?"

        except Exception as e:
            job.status = "failed"
            job.error = str(e)

        job.completed_at = datetime.now()
        return job

    def _parse_metrics(self, stdout: str) -> Dict[str, float]:
        """
        Parse metrics from TerraTorch output.

        Args:
            stdout: Standard output from TerraTorch

        Returns:
            Dict of metric names to values
        """
        metrics = {}

        # Common patterns in TerraTorch/PyTorch Lightning output
        patterns = [
            # "test_accuracy: 0.9234"
            (r"test_accuracy[:\s]+([0-9.]+)", "accuracy"),
            # "test_f1: 0.8912"
            (r"test_f1[:\s]+([0-9.]+)", "f1"),
            # "test_miou: 0.7543"
            (r"test_miou[:\s]+([0-9.]+)", "miou"),
            # "test_loss: 0.2341"
            (r"test_loss[:\s]+([0-9.]+)", "loss"),
            # "accuracy=0.9234"
            (r"accuracy[=:\s]+([0-9.]+)", "accuracy"),
            # "Accuracy: 92.34%"
            (r"[Aa]ccuracy[:\s]+([0-9.]+)%?", "accuracy"),
            # "F1 Score: 0.89"
            (r"[Ff]1[_ ][Ss]core[:\s]+([0-9.]+)", "f1"),
            # "mIoU: 0.75"
            (r"mIoU[:\s]+([0-9.]+)", "miou"),
        ]

        for pattern, metric_name in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match and metric_name not in metrics:
                try:
                    value = float(match.group(1))
                    # Convert percentage to decimal if > 1
                    if metric_name in ["accuracy", "f1"] and value > 1:
                        value = value / 100
                    metrics[metric_name] = value
                except ValueError:
                    pass

        return metrics

    def check_gpu_available(self, min_memory_gb: int = 8) -> bool:
        """
        Check if GPU has sufficient memory.

        Args:
            min_memory_gb: Minimum required GPU memory in GB

        Returns:
            True if GPU is available with sufficient memory
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return False

            memory_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
            )
            return memory_gb >= min_memory_gb

        except ImportError:
            return False

    def _parse_metrics_from_csv(self, job_output_dir: Path) -> Dict[str, float]:

        metrics_file = (
                job_output_dir
                / "lightning_logs"
                / "version_0"
                / "metrics.csv"
        )

        if not metrics_file.exists():
            return {}

        df = pd.read_csv(metrics_file)

        if df.empty:
            return {}

        last_row = df.iloc[-1]
        metrics = {}

        mapping = {
            "test/Accuracy": "accuracy",
            "test/F1_Score": "f1",
            "test/Precision": "precision",
            "test/Recall": "recall",
            "test/loss": "loss",
        }

        for csv_name, api_name in mapping.items():
            if csv_name in last_row and not pd.isna(last_row[csv_name]):
                metrics[api_name] = float(last_row[csv_name])

        return metrics