"""SLURM-based job runner for remote benchmark execution via SSH."""

import logging
import re
import textwrap
import time
import yaml
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, Optional

import paramiko

from REMSA.utils.job_runner import BenchmarkJob

logger = logging.getLogger(__name__)

# SLURM states that indicate the job is still running
_RUNNING_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"}
# SLURM states that indicate completion (successful or not)
_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


class SlurmJobRunner:
    """Executes benchmark jobs on a remote SLURM cluster via SSH."""

    def __init__(
        self,
        output_dir: Path,
        host: str,
        user: str,
        password: str,
        work_dir: str = "/tmp/benchmarks",
        partition: str = "gpu",
        account: Optional[str] = None,
        conda_env: Optional[str] = None,
        conda_base: Optional[str] = None,
        modules: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.host = host
        self.user = user
        self.password = password
        self.work_dir = PurePosixPath(work_dir)
        self.partition = partition
        self.account = account
        self.conda_env = conda_env
        self.conda_base = conda_base
        self.modules = modules.split() if modules else []

    def _connect(self) -> paramiko.SSHClient:
        """Create and return an SSH connection."""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.host, username=self.user, password=self.password)
        return ssh

    def _ssh_exec(self, ssh: paramiko.SSHClient, command: str) -> str:
        """Execute a command over SSH and return stdout. Raises on non-zero exit."""
        logger.debug("SSH exec: %s", command)
        _, stdout, stderr = ssh.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if exit_code != 0:
            raise RuntimeError(f"Remote command failed (exit {exit_code}): {err or out}")
        return out

    def run_job(
        self,
        job: BenchmarkJob,
        mode: str = "test",
        checkpoint_path: Optional[Path] = None,
        checkpoint_save_path: Optional[Path] = None,
    ) -> BenchmarkJob:
        """Execute a benchmark job on the remote SLURM cluster."""
        job.status = "running"
        job.started_at = datetime.now()

        ssh = None
        try:
            ssh = self._connect()
            sftp = ssh.open_sftp()

            # Create remote job directory
            remote_job_dir = self.work_dir / job.job_id
            remote_data_dir = remote_job_dir / "data"
            self._ssh_exec(ssh, f"mkdir -p {remote_data_dir}")

            # Load local config, patch dataset path to remote location, upload via SFTP
            with open(job.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            try:
                cfg["data"]["dict_kwargs"]["root"] = str(remote_data_dir)
            except (KeyError, TypeError):
                logger.warning("Could not patch data.dict_kwargs.root; uploading config as-is")
            remote_config_path = remote_job_dir / Path(job.config_path).name
            with sftp.open(str(remote_config_path), "w") as rf:
                rf.write(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            logger.info("Uploaded patched config to %s:%s", self.host, remote_config_path)

            # Upload checkpoint if provided
            remote_ckpt_path = None
            if mode == "test" and checkpoint_path:
                remote_ckpt_path = remote_job_dir / Path(checkpoint_path).name
                sftp.put(str(checkpoint_path), str(remote_ckpt_path))

            # Generate sbatch script
            remote_output_dir = remote_job_dir / "output"
            sbatch_content = self._generate_sbatch_script(
                job_id=job.job_id,
                remote_config_path=str(remote_config_path),
                remote_output_dir=str(remote_output_dir),
                remote_job_dir=str(remote_job_dir),
                mode=mode,
                checkpoint_path=str(remote_ckpt_path) if remote_ckpt_path else None,
            )

            # Upload sbatch script
            sbatch_path = remote_job_dir / "run.sh"
            with sftp.open(str(sbatch_path), "w") as f:
                f.write(sbatch_content)

            # Submit job
            submit_output = self._ssh_exec(ssh, f"sbatch {sbatch_path}")
            slurm_job_id = self._parse_sbatch_output(submit_output)
            logger.info("Submitted SLURM job %s for benchmark %s", slurm_job_id, job.job_id)

            # Poll until terminal state
            final_state = self._poll_slurm_status(ssh, slurm_job_id)
            logger.info("SLURM job %s finished with state: %s", slurm_job_id, final_state)

            # Retrieve output
            stdout_content = self._retrieve_output(ssh, sftp, remote_job_dir, slurm_job_id)

            if final_state == "COMPLETED":
                job.status = "completed"
                job.result = {
                    "stdout": stdout_content,
                    "stderr": "",
                    "metrics": self._parse_metrics(stdout_content),
                    "output_dir": str(self.results_dir / job.job_id),
                    "slurm_job_id": slurm_job_id,
                }
                # Download fine-tuned checkpoint to stable local path
                if mode == "fit" and checkpoint_save_path is not None:
                    self._download_checkpoint(ssh, sftp, str(remote_output_dir), checkpoint_save_path)
            else:
                job.status = "failed"
                job.error = f"SLURM job ended with state: {final_state}"
                job.result = {
                    "stdout": stdout_content,
                    "stderr": "",
                    "metrics": {},
                    "slurm_job_id": slurm_job_id,
                }

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.exception("SLURM job execution failed for %s", job.job_id)

        finally:
            if ssh:
                ssh.close()

        job.completed_at = datetime.now()
        return job

    def _download_checkpoint(
        self,
        ssh: paramiko.SSHClient,
        sftp: paramiko.SFTPClient,
        remote_output_dir: str,
        save_path: Path,
    ) -> None:
        """Find and download the best checkpoint from the remote cluster."""
        try:
            remote_ckpt = self._ssh_exec(
                ssh,
                f'find {remote_output_dir} -name "*.ckpt" -path "*/checkpoints/*" | sort | tail -1',
            ).strip()
            if not remote_ckpt:
                logger.warning("No checkpoint found in %s; skipping download", remote_output_dir)
                return
            save_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_ckpt, str(save_path))
            logger.info("Downloaded checkpoint %s → %s", remote_ckpt, save_path)
        except Exception:
            logger.exception("Failed to download checkpoint from %s", remote_output_dir)

    def _generate_sbatch_script(
        self,
        job_id: str,
        remote_config_path: str,
        remote_output_dir: str,
        remote_job_dir: str,
        mode: str,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        """Generate an sbatch submission script."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=bench_{job_id}",
            f"#SBATCH --partition={self.partition}",
        ]
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        lines += [
            "#SBATCH --gres=gpu:1",
            f"#SBATCH --output={remote_job_dir}/slurm_%j.out",
            f"#SBATCH --error={remote_job_dir}/slurm_%j.err",
            "",
        ]

        # Module loads
        for mod in self.modules:
            lines.append(f"module load {mod}")
        if self.modules:
            lines.append("")

        # Conda activation
        if self.conda_env:
            if self.conda_base:
                lines.append(f"source {self.conda_base}/etc/profile.d/conda.sh")
            else:
                lines.append('eval "$(conda shell.bash hook)"')
            lines.append(f"conda activate {self.conda_env}")
            lines.append("")

        # terratorch command
        cmd = f"terratorch {mode} --config {remote_config_path} --trainer.default_root_dir {remote_output_dir}"
        if mode == "test" and checkpoint_path:
            cmd += f" --ckpt_path {checkpoint_path}"
        lines.append(cmd)

        # After fit, automatically run test using the best checkpoint
        # so we get a metrics table in stdout
        if mode == "fit":
            lines.append("")
            lines.append("# Auto-run test after training to collect metrics")
            lines.append(
                f'CKPT=$(find {remote_output_dir} -name "*.ckpt" -path "*/checkpoints/*" | sort | tail -1)'
            )
            lines.append('if [ -n "$CKPT" ]; then')
            lines.append(
                f"  terratorch test --config {remote_config_path}"
                f" --trainer.default_root_dir {remote_output_dir}"
                f' --ckpt_path "$CKPT"'
            )
            lines.append("fi")

        return "\n".join(lines) + "\n"

    def _parse_sbatch_output(self, output: str) -> str:
        """Parse SLURM job ID from sbatch output like 'Submitted batch job 12345'."""
        match = re.search(r"Submitted batch job (\d+)", output)
        if not match:
            raise RuntimeError(f"Could not parse SLURM job ID from: {output}")
        return match.group(1)

    def _poll_slurm_status(
        self,
        ssh: paramiko.SSHClient,
        slurm_job_id: str,
        timeout: int = 3600,
        interval: int = 10,
    ) -> str:
        """Poll sacct until the job reaches a terminal state."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                out = self._ssh_exec(
                    ssh,
                    f"sacct -j {slurm_job_id} --format=State --noheader -P",
                )
                # sacct can return multiple lines (one per job step); take the first
                states = [s.strip() for s in out.splitlines() if s.strip()]
                if states:
                    state = states[0].split("+")[0]  # strip trailing '+' from e.g. "CANCELLED+"
                    if state in _TERMINAL_STATES:
                        return state
                    if state not in _RUNNING_STATES:
                        logger.warning("Unknown SLURM state: %s", state)
            except RuntimeError:
                # sacct may not return results immediately after submission
                pass

            time.sleep(interval)

        raise TimeoutError(f"SLURM job {slurm_job_id} did not complete within {timeout}s")

    def _retrieve_output(
        self,
        ssh: paramiko.SSHClient,
        sftp: paramiko.SFTPClient,
        remote_job_dir: PurePosixPath,
        slurm_job_id: str,
    ) -> str:
        """Retrieve stdout from the SLURM output file."""
        remote_stdout = str(remote_job_dir / f"slurm_{slurm_job_id}.out")
        try:
            with sftp.open(remote_stdout, "r") as f:
                return f.read().decode()
        except FileNotFoundError:
            logger.warning("SLURM output file not found: %s", remote_stdout)
            return ""

    def _parse_metrics(self, stdout: str) -> Dict[str, float]:
        """Parse metrics from TerraTorch/PyTorch Lightning table output.

        Handles the rich table format:
        │       test/Accuracy       │    0.10176033526659012    │
        """
        metrics = {}

        # Match rows in the PL rich table: │  metric_name  │  value  │
        # Handles both test/ and val/ prefixes (fit mode logs val metrics)
        table_pattern = re.compile(
            r"│\s+((?:test|val)/\S+)\s+│\s+([0-9.eE+-]+)\s+│"
        )

        # Map from PL metric names to short display names.
        # Only pick up the summary metrics, skip per-class rows.
        # test/ metrics take priority over val/ if both exist.
        metric_map = {
            "test/Accuracy": "accuracy",
            "test/Accuracy_Micro": "accuracy_micro",
            "test/F1_Score": "f1_score",
            "test/Precision": "precision",
            "test/Recall": "recall",
            "test/loss": "loss",
            "test/mIoU": "miou",
            "val/Accuracy": "accuracy",
            "val/Accuracy_Micro": "accuracy_micro",
            "val/F1_Score": "f1_score",
            "val/Precision": "precision",
            "val/Recall": "recall",
            "val/loss": "loss",
            "val/mIoU": "miou",
        }

        for match in table_pattern.finditer(stdout):
            name = match.group(1)
            short = metric_map.get(name)
            if short and short not in metrics:
                try:
                    metrics[short] = float(match.group(2))
                except ValueError:
                    pass

        return metrics

    def check_gpu_available(self, min_memory_gb: int = 8) -> bool:
        """Remote cluster is assumed to have GPUs available."""
        return True
