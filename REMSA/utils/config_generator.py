"""Generate TerraTorch YAML configuration files."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


class TerraTorchConfigGenerator:
    """Generate TerraTorch YAML configuration files."""

    TASK_CLASS_MAP = {
        "classification": "terratorch.tasks.ClassificationTask",
        "multi_label": "terratorch.tasks.MultiLabelClassificationTask",
        "segmentation": "terratorch.tasks.SemanticSegmentationTask",
    }

    def create_config(
        self,
        model_family: str,
        variant: str,
        backbone: str,
        dataset_config: Dict[str, Any],
        output_dir: Path,
        batch_size: int = 8,
        num_workers: int = 4,
        max_epochs: int = 10,
        learning_rate: float = 1e-4,
        mode: str = "test",
        model_factory: str = "EncoderDecoderFactory",
        band_requirements: Optional[List[str]] = None,
        extra_model_args: Optional[Dict[str, Any]] = None,
        checkpoint_save_dir: Optional[Path] = None,
    ) -> Path:
        """
        Create TerraTorch YAML config file.

        Args:
            model_family: Model family name (e.g., 'prithvi')
            variant: Model variant name (e.g., 'prithvi_100m')
            backbone: TerraTorch backbone name
            dataset_config: Dataset configuration from DatasetRegistry
            output_dir: Directory for output files
            batch_size: Batch size for training/evaluation
            num_workers: Number of data loader workers
            max_epochs: Maximum training epochs
            learning_rate: Learning rate
            mode: "fit" for training, "test" for evaluation only

        Returns:
            Path to created YAML config file

        Raises:
            ValueError: If task type is unknown
        """
        # Determine task class
        task_type = dataset_config["task"]
        task_class = self.TASK_CLASS_MAP.get(task_type)
        if not task_class:
            raise ValueError(f"Unknown task type: {task_type}")

        # Build model args with appropriate decoder for task type
        model_args = {
            "backbone": backbone,
            "num_classes": dataset_config["num_classes"],
        }

        # Set decoder based on task type
        if task_type == "segmentation":
            if mode == "linear_probe":
                model_args.update({
                    "decoder": "FCNDecoder",
                    "decoder_channels": 256,
                })
            elif mode == "finetune":  # finetune
                model_args.update({
                    "decoder": "UNetDecoder",
                    "decoder_channels": [512, 256, 128, 64],
                    "head_channel_list": [256],
                    "head_dropout": 0.1,
                })
            else:
                model_args.update({
                    "decoder": "FCNDecoder",
                    "decoder_channels": 256,
                })

            model_args.update({
                "backbone_pretrained": True,
                "necks": [
                    {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                    {"name": "ReshapeTokensToImage"},
                    {"name": "LearnedInterpolateToPyramidal"},
                ],
            })
        else:
            # For classification and multi-label, use identity decoder
            # which passes features to classification head
            model_args["decoder"] = "terratorch_IdentityDecoder"

        # Resolve band requirements: variant-level overrides family-level
        effective_bands = band_requirements

        # Merge any extra model args provided by the benchmarker
        # (each factory/backbone needs different args like model_bands,
        #  in_channels, bands, backbone_bands, etc.)
        if extra_model_args:
            model_args.update(extra_model_args)

        # Build data module (with band selection if needed)
        data_module = self._build_data_module(
            model_family,
            dataset_config,
            output_dir,
            batch_size,
            num_workers,
            band_requirements=effective_bands,
        )

        # Determine loss function (using short names that terratorch accepts)
        if task_type == "segmentation":
            loss = "dice"
        elif task_type == "multi_label":
            loss = "bce"
        else:
            loss = "ce"

        model_init_args = {
            "model_factory": model_factory,
            "model_args": model_args,
            "loss": loss,
        }

        if task_type == "segmentation":
            if mode in ["linear_probe", "test"]:
                model_init_args["freeze_backbone"] = True
                model_init_args["freeze_decoder"] = False
            else:
                model_init_args["freeze_backbone"] = False
                model_init_args["freeze_decoder"] = False

        weight_decay = 0.1 if mode == "finetune" else 0.01

        config = {
            "seed_everything": 42,
            "model": {
                "class_path": task_class,
                "init_args": model_init_args,
            },
            "data": data_module,
            "trainer": {
                "accelerator": "auto",
                "strategy": "auto",
                "devices": "auto",
                "num_nodes": 1,
                "precision": "16-mixed",
                "max_epochs": max_epochs,
                "check_val_every_n_epoch": 1,
                "log_every_n_steps": 10,
                "enable_checkpointing": True,
            },
            "optimizer": {
                "class_path": "torch.optim.AdamW",
                "init_args": {
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                },
            },
            "lr_scheduler": {
                "class_path": "ReduceLROnPlateau",
                "init_args": {
                    "monitor": "val/loss",
                    "patience": 5,
                    "factor": 0.5,
                },
            },
        }

        trainer = config["trainer"]

        trainer["logger"] = {
            "class_path": "lightning.pytorch.loggers.CSVLogger",
            "init_args": {
                "save_dir": str(output_dir / f"{variant}_{mode}_results"),
                "name": "logs",
            },
        }

        trainer["callbacks"] = [
            {"class_path": "RichProgressBar"},
            {
                "class_path": "LearningRateMonitor",
                "init_args": {"logging_interval": "epoch"},
            },
            {
                "class_path": "ModelCheckpoint",
                "init_args": {
                    "dirpath": str(output_dir / f"{variant}_{mode}_results/checkpoints"),
                    "filename": "best-model-{epoch:02d}-{val/loss:.3f}",
                    "monitor": "val/loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "save_last": True,
                },
            },
        ]

        if task_type == "segmentation" and mode == "finetune":
            model_init_args["aux_loss"] = {"aux_head": 1.0}

        # inject ModelCheckpoint callback when a stable save dir is provided
        if checkpoint_save_dir is not None:
            print(checkpoint_save_dir)
            checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
            config["trainer"]["callbacks"].append({
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": str(checkpoint_save_dir),
                    "save_last": True,
                    "filename": "checkpoint",
                }
            })

        if task_type == "segmentation":
            model_init_args["ignore_index"] = -1

        if mode == "test":
            config["trainer"]["max_epochs"] = 0
            config["trainer"]["limit_train_batches"] = 0
            config["trainer"]["limit_val_batches"] = 1

        # Save config (use variant_dataset naming since variant includes family)
        config_path = (
            output_dir
            / "configs"
            / f"{variant}_{dataset_config['name']}.yaml"
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return config_path

    def _build_data_module(
        self,
        model_family: str,
        dataset_config: Dict[str, Any],
        output_dir: Path,
        batch_size: int,
        num_workers: int,
        band_requirements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build data module configuration.

        Args:
            model_family: Model family name for band selection
            dataset_config: Dataset configuration from registry
            output_dir: Directory for data storage
            batch_size: Batch size
            num_workers: Number of data loader workers
            band_requirements: Resolved HLS band names to select

        Returns:
            Data module configuration dictionary

        Raises:
            ValueError: If no data module configuration available
        """
        # Get dataset bands to load using resolved band requirements
        all_bands = dataset_config.get("all_hls_band_names")
        output_bands = self._get_dataset_bands_for(
            band_requirements,
            dataset_config
        )

        if dataset_config.get("torchgeo"):
            dm_path = (
                f"torchgeo.datamodules.{dataset_config['data_module']}"
            )
            # batch_size and num_workers are explicit params;
            # root, download, bands go through **kwargs so must
            # use dict_kwargs for jsonargparse/LightningCLI.
            init_args = {
                "batch_size": batch_size,
                "num_workers": num_workers,
            }
            dict_kwargs = {
                "root": str(output_dir / "data"),
                "download": True,
            }
            if output_bands:
                dict_kwargs["bands"] = output_bands

            return {
                "class_path": dm_path,
                "init_args": init_args,
                "dict_kwargs": dict_kwargs,
            }
        elif dataset_config.get("huggingface"):
            return {
                "class_path": (
                    "terratorch.datamodules.HuggingFaceDataModule"
                ),
                "init_args": {
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                },
                "dict_kwargs": {
                    "dataset_name": dataset_config["huggingface"],
                },
            }
        elif dataset_config["data_module"] == "GenericNonGeoSegmentationDataModule":
            if not all_bands:
                raise ValueError(
                    f"{dataset_config['name']} missing 'all_hls_band_names'"
                )
            root = dataset_config["data_root"]

            init_args = {
                "batch_size": batch_size,
                "num_workers": num_workers,

                # Image roots
                "train_data_root": f"{root}/data/S2L1CHand",
                "val_data_root": f"{root}/data/S2L1CHand",
                "test_data_root": f"{root}/data/S2L1CHand",

                # Label roots
                "train_label_data_root": f"{root}/data/LabelHand",
                "val_label_data_root": f"{root}/data/LabelHand",
                "test_label_data_root": f"{root}/data/LabelHand",

                # Splits
                "train_split": f"{root}/splits/flood_train_data.txt",
                "val_split": f"{root}/splits/flood_valid_data.txt",
                "test_split": f"{root}/splits/flood_test_data.txt",

                "img_grep": "*_S2Hand.tif",
                "label_grep": "*_LabelHand.tif",

                "no_label_replace": -1,
                "no_data_replace": 0,

                "num_classes": dataset_config["num_classes"],

                "train_transform": [
                    {"class_path": "albumentations.RandomCrop",
                     "init_args": {"height": 256, "width": 256}},
                    {"class_path": "albumentations.D4"},
                    {"class_path": "albumentations.pytorch.transforms.ToTensorV2"}
                ],
                "val_transform": [
                    {"class_path": "albumentations.CenterCrop",
                     "init_args": {"height": 256, "width": 256}},
                    {"class_path": "albumentations.pytorch.transforms.ToTensorV2"}
                ],
                "test_transform": [
                    {"class_path": "albumentations.CenterCrop",
                     "init_args": {"height": 256, "width": 256}},
                    {"class_path": "albumentations.pytorch.transforms.ToTensorV2"}
                ],
                "drop_last": False,
            }

            if band_requirements:
                # stay in HLS space (like your working YAML)
                dataset_bands = all_bands
                output_bands = band_requirements
            else:
                dataset_bands = all_bands
                output_bands = all_bands

            init_args["dataset_bands"] = dataset_bands
            init_args["output_bands"] = output_bands

            # normalization
            norm = dataset_config.get("normalization")

            if norm:
                init_args["means"] = norm["means"]
                init_args["stds"] = norm["stds"]

            # rgb indices
            rgb_indices = dataset_config.get("rgb_indices")
            if rgb_indices:
                init_args["rgb_indices"] = rgb_indices

            return {
                "class_path": "terratorch.datamodules.GenericNonGeoSegmentationDataModule",
                "init_args": init_args,
            }

        else:
            raise ValueError(
                f"No data module configuration for "
                f"{dataset_config['name']}"
            )

    def _get_dataset_bands_for(
        self,
        band_requirements: Optional[List[str]],
        dataset_config: Dict[str, Any],
    ) -> Optional[List[str]]:
        """
        Get the list of dataset band names to load.

        Args:
            band_requirements: HLS band names the model requires, or None
            dataset_config: Dataset configuration from registry

        Returns:
            List of dataset band names (e.g., ['B02', 'B03', ...]) to load,
            or None if all bands should be loaded.
        """
        if not band_requirements:
            return None

        hls_band_mapping = dataset_config.get("hls_band_mapping")
        if not hls_band_mapping:
            return None

        dataset_bands = []
        for hls_band in band_requirements:
            if hls_band not in hls_band_mapping:
                raise ValueError(
                    f"Dataset '{dataset_config['name']}' is missing "
                    f"band mapping for {hls_band}"
                )
            dataset_bands.append(hls_band_mapping[hls_band])

        return dataset_bands
