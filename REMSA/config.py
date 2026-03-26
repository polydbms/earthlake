import yaml
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "remsa_config.yaml")

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

if "OPENAI_API_KEY" in os.environ:
    print(f"Setting OPENAI_API_KEY to {os.environ['OPENAI_API_KEY'][:15]}***")
    config["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

# SLURM / benchmark executor settings from environment
for key in [
    "SLURM_HOST", "SLURM_USER", "SLURM_PASSWORD", "SLURM_WORK_DIR",
    "SLURM_PARTITION", "SLURM_ACCOUNT", "SLURM_CONDA_ENV", "SLURM_CONDA_BASE", "SLURM_MODULES", "BENCHMARK_EXECUTOR",
    "MODEL_WEIGHTS_DIR",
]:
    if key in os.environ:
        config[key] = os.environ[key]