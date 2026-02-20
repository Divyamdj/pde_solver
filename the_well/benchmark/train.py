# the_well/benchmark/train.py

import logging
import os.path as osp

import hydra
import torch
import torch.distributed as dist
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

from the_well.benchmark.trainer import Trainer
from the_well.benchmark.trainer.utils import set_master_config
from the_well.benchmark.utils.experiment_utils import configure_experiment
from the_well.data.multi_datamodule import MultiWellDataModule
from the_well.data.normalization import ZScoreNormalization
from the_well.data import WellDataModule

logger = logging.getLogger("the_well")
logger.setLevel(level=logging.DEBUG)

# Retrieve configuration for hydra
CONFIG_DIR = osp.join(osp.dirname(__file__), "configs")
CONFIG_NAME = "config"
CONFIG_PATH = osp.join(CONFIG_DIR, f"{CONFIG_NAME}.yaml")
assert osp.isfile(CONFIG_PATH), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")


def train(
    cfg: DictConfig,
    experiment_folder: str,
    checkpoint_folder: str,
    artifact_folder: str,
    viz_folder: str,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 1,
    local_rank: int = 1,
):
    """Instantiate the different objects required for training and run the training loop."""
    validation_mode = cfg.validation_mode
    
    # Check if using multi-dataset training
    is_multi_dataset = hasattr(cfg.data, "train_datasets")
    
    if is_multi_dataset:
        logger.info("Using multi-dataset training mode")

        datamodule = MultiWellDataModule(
            well_base_path=cfg.data.well_base_path,
            train_datasets=cfg.data.train_datasets,
            val_datasets=cfg.data.val_datasets,
            test_datasets=cfg.data.test_datasets,
            batch_size=cfg.data.batch_size,
            data_workers=cfg.data_workers,
            n_steps_input=cfg.data.n_steps_input,
            n_steps_output=cfg.data.n_steps_output,
            use_normalization=cfg.data.get("use_normalization", True),
            target_hw=tuple(cfg.data.get("target_hw", [128, 384])),
        )
        datamodule.setup()

        logger.info(f"data_workers = {cfg.data_workers}")

        
        dset_metadata = datamodule.train_dataset.metadata
        # For multi-dataset, channels are padded to c_max
        n_input_fields = cfg.data.n_steps_input * datamodule.train_dataset.c_max
        n_output_fields = cfg.data.n_steps_output * datamodule.train_dataset.c_max
        
        logger.info(f"Multi-dataset training:")
        logger.info(f"  Train datasets: {cfg.data.train_datasets}")
        logger.info(f"  Val datasets: {cfg.data.val_datasets}")
        logger.info(f"  Test datasets: {cfg.data.test_datasets}")
        logger.info(f"  Total channels (c_max): {datamodule.train_dataset.c_max}")
        logger.info(f"  Input fields: {n_input_fields} (= {cfg.data.n_steps_input} timesteps × {datamodule.train_dataset.c_max} channels)")
        logger.info(f"  Output fields: {n_output_fields}")
    else:
        logger.info("Using single-dataset training mode")
        logger.info(f"Instantiate datamodule {cfg.data._target_}")
        datamodule: WellDataModule = instantiate(
            cfg.data, world_size=world_size, rank=rank, data_workers=cfg.data_workers
        )
        dset_metadata = datamodule.train_dataset.metadata
        # For single dataset, calculate based on metadata
        n_input_fields = (
            cfg.data.n_steps_input * dset_metadata.n_fields
            + dset_metadata.n_constant_fields
        )
        n_output_fields = dset_metadata.n_fields

    logger.info(
        f"Instantiate model {cfg.model._target_}",
    )
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_spatial_dims=dset_metadata.n_spatial_dims,
        spatial_resolution=dset_metadata.spatial_resolution,
        dim_in=n_input_fields,
        dim_out=n_output_fields,
    )
    summary(model, depth=5)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
    else:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model = model.to(device)


    logger.info(f"device = {device}")


    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    if not validation_mode:
        optimizer: torch.optim.Optimizer = instantiate(
            cfg.optimizer, params=model.parameters()
        )
    else:
        optimizer = None

    if hasattr(cfg, "lr_scheduler") and not validation_mode:
        # Instantiate LR scheduler
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=cfg.trainer.epochs,
            warmup_start_lr=cfg.optimizer.lr * 0.1,
            eta_min=cfg.optimizer.lr * 0.1,
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None
    # Print final config, but also log it to experiment directory.
    logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Instantiate trainer {cfg.trainer._target_}")

    print("--------------------------------------------------", datamodule)
    
    # For multi-dataset mode, construct Trainer directly to avoid Hydra deserialization issues
    if is_multi_dataset:
        trainer: Trainer = Trainer(
            checkpoint_folder=checkpoint_folder,
            artifact_folder=artifact_folder,
            viz_folder=viz_folder,
            formatter=cfg.trainer.get("formatter", "channels_first_default"),
            model=model,
            datamodule=datamodule,
            optimizer=optimizer,
            loss_fn=instantiate(cfg.trainer.loss_fn),
            epochs=cfg.trainer.epochs,
            checkpoint_frequency=cfg.trainer.checkpoint_frequency,
            val_frequency=cfg.trainer.val_frequency,
            rollout_val_frequency=cfg.trainer.rollout_val_frequency,
            max_rollout_steps=cfg.trainer.max_rollout_steps,
            short_validation_length=cfg.trainer.short_validation_length,
            make_rollout_videos=cfg.trainer.make_rollout_videos,
            num_time_intervals=cfg.trainer.num_time_intervals,
            lr_scheduler=lr_scheduler,
            device=device,
            is_distributed=is_distributed,
            enable_amp=cfg.get("enable_amp", False),
        )
        # Disable verbose tqdm progress bars for cleaner logs
        trainer.show_progress = False
    else:
        trainer: Trainer = instantiate(
            cfg.trainer,
            checkpoint_folder=checkpoint_folder,
            artifact_folder=artifact_folder,
            viz_folder=viz_folder,
            model=model,
            datamodule=datamodule,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            is_distributed=is_distributed,
        )
        # Disable verbose tqdm progress bars for cleaner logs
        trainer.show_progress = False
    if validation_mode:
        trainer.validate()
    else:
        # Save config to directory folder
        with open(osp.join(experiment_folder, "extended_config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
        trainer.train()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    # Torch optimization settings
    torch.backends.cudnn.benchmark = (
        True  # If input size is fixed, this will usually the computation faster
    )
    torch.set_float32_matmul_precision("high")  # Use TF32 when supported
    # Normal things
    (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
    ) = configure_experiment(cfg, logger)

    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Determine wandb group based on dataset configuration
    if hasattr(cfg.data, "train_datasets"):
        # Multi-dataset training
        wandb_group = f"multi_{'_'.join(cfg.data.train_datasets[:2])}"
    else:
        # Single dataset training
        wandb_group = f"{cfg.data.well_dataset_name}"
    
    # Initiate wandb logging
    wandb_logged_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_logged_cfg["experiment_folder"] = experiment_folder
    wandb.init(
        dir=experiment_folder,
        project=cfg.wandb_project_name,
        group=wandb_group,
        config=wandb_logged_cfg,
        name=experiment_name,
        resume=True,
    )

    # Retrieve multiple processes context to setup DDP
    is_distributed, world_size, rank, local_rank = (
        False,
        1,
        0,
        0,
    )  # get_distrib_config()
    # is_distributed = is_distributed and world_size > 1

    logger.info(f"Distributed training: {is_distributed}")
    if is_distributed:
        set_master_config()
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
    train(
        cfg,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
        is_distributed,
        world_size,
        rank,
        local_rank,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
