import logging
import os.path as osp
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from the_well.benchmark.metrics import MSE
from the_well.benchmark.optim.schedulers import LinearWarmupCosineAnnealingLR
from the_well.benchmark.trainer import Trainer
from the_well.benchmark.utils.experiment_utils import configure_experiment
from the_well.benchmark.models.unet_classic_conditioned import UNetClassicConditioned
from the_well.data.multi_datamodule import MultiWellDataModule

logger = logging.getLogger("the_well")
logger.setLevel(level=logging.DEBUG)

# ---------------------------------------------------------------------
# Hydra config location
# ---------------------------------------------------------------------
# Retrieve configuration for hydra
CONFIG_DIR = osp.join(osp.dirname(__file__), "configs")
CONFIG_NAME = "config"
CONFIG_PATH = osp.join(CONFIG_DIR, f"{CONFIG_NAME}.yaml")
assert osp.isfile(CONFIG_PATH), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")

def _ensure_nested_cfg(cfg: DictConfig) -> DictConfig:
    """
    Accepts either:
      A) Nested config: cfg.data.*, cfg.model.*, cfg.optimizer.*, cfg.trainer.*
      B) Flat config: well_base_path, train_datasets, lr, epochs, init_features, ...

    Returns a nested cfg compatible with train.py-style code.
    """

    # If it already has the structure, do nothing
    if hasattr(cfg, "data") and hasattr(cfg, "trainer") and hasattr(cfg, "optimizer") and hasattr(cfg, "model"):
        return cfg

    logger.warning(
        "Config appears to be FLAT. Converting to train.py-style nested config automatically."
    )

    # Build a nested config from the flat keys
    nested = OmegaConf.create(
        {
            # experiment meta
            "name": cfg.get("name", "multi_pde_run"),
            "experiment_dir": cfg.get("experiment_dir", "experiments"),
            "auto_resume": cfg.get("auto_resume", False),
            "folder_override": cfg.get("folder_override", ""),
            "checkpoint_override": cfg.get("checkpoint_override", ""),
            "config_override": cfg.get("config_override", ""),
            "validation_mode": cfg.get("validation_mode", False),
            "wandb_project_name": cfg.get("wandb_project_name", "the_well_extended_again"),
            "data_workers": cfg.get("data_workers", 8),
            "enable_amp": cfg.get("enable_amp", False),

            # data block
            "data": {
                "well_base_path": cfg.well_base_path,
                "train_datasets": cfg.train_datasets,
                "val_datasets": cfg.val_datasets,
                "test_datasets": cfg.test_datasets,
                "batch_size": cfg.batch_size,
                "n_steps_input": cfg.n_steps_input,
                "n_steps_output": cfg.n_steps_output,
                "target_hw": cfg.target_hw,
                "use_normalization": cfg.get("use_normalization", True),
                "min_dt_stride": cfg.get("min_dt_stride", 1),
                "max_dt_stride": cfg.get("max_dt_stride", 1),
            },

            # model block
            "model": {
                "init_features": cfg.get("init_features", 48),
                "cond_channels": cfg.get("cond_channels", 16),
                "emb_dim": cfg.get("emb_dim", 384),
            },

            # optimizer block
            "optimizer": {
                "lr": cfg.get("lr", 1e-3),
                "weight_decay": cfg.get("weight_decay", 1e-4),
            },

            # scheduler block
            "lr_scheduler": {
                "warmup_epochs": cfg.get("warmup_epochs", 5),
            },

            # trainer block
            "trainer": {
                "epochs": cfg.get("epochs", 1),
                "checkpoint_frequency": cfg.get("checkpoint_frequency", 10),
                "val_frequency": cfg.get("val_frequency", 1),
                "rollout_val_frequency": cfg.get("rollout_val_frequency", 2),
                "short_validation_length": cfg.get("short_validation_length", 10),
                "max_rollout_steps": cfg.get("max_rollout_steps", 100),
                "num_time_intervals": cfg.get("num_time_intervals", 5),
                "make_rollout_videos": cfg.get("make_rollout_videos", False),
                "checkpoint_path": cfg.get("checkpoint_path", ""),
            },
        }
    )
    return nested


def train(
    cfg: DictConfig,
    experiment_folder: str,
    checkpoint_folder: str,
    artifact_folder: str,
    viz_folder: str,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
):
    """
    PDE multi-dataset training loop.
    Mirrors train.py as closely as possible.
    """

    cfg = _ensure_nested_cfg(cfg)
    validation_mode = cfg.validation_mode

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------
    logger.info("Using multi-dataset PDE training mode")

    datamodule = MultiWellDataModule(
        well_base_path=cfg.data.well_base_path,
        train_datasets=cfg.data.train_datasets,
        val_datasets=cfg.data.val_datasets,
        test_datasets=cfg.data.test_datasets,
        batch_size=cfg.data.batch_size,
        data_workers=cfg.data_workers,
        n_steps_input=cfg.data.n_steps_input,
        n_steps_output=cfg.data.n_steps_output,
        target_hw=tuple(cfg.data.target_hw),
        use_normalization=cfg.data.get("use_normalization", True),
        min_dt_stride=cfg.data.get("min_dt_stride", 1),
        max_dt_stride=cfg.data.get("max_dt_stride", 1),
    )
    datamodule.setup()

    # -----------------------------------------------------------------
    # CHANNEL COUNTS
    # -----------------------------------------------------------------
    # MultiWellDataset pads channels to c_max
    C = datamodule.train_dataset.c_max
    Ti = cfg.data.n_steps_input
    To = cfg.data.n_steps_output

    n_input_fields = Ti * C
    n_output_fields = To * C

    logger.info("Multi-dataset training:")
    logger.info(f"  Train datasets: {cfg.data.train_datasets}")
    logger.info(f"  Val datasets:   {cfg.data.val_datasets}")
    logger.info(f"  Test datasets:  {cfg.data.test_datasets}")
    logger.info(f"  Total channels (c_max): {C}")
    logger.info(f"  Input fields:  {n_input_fields} (= {Ti} timesteps × {C} channels)")
    logger.info(f"  Output fields: {n_output_fields} (= {To} timesteps × {C} channels)")
    logger.info(f"  Target resolution: {cfg.data.target_hw}")

    # -----------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------
    logger.info("Instantiate UNetClassicConditioned")

    model = UNetClassicConditioned(
        dim_in=n_input_fields,
        dim_out=n_output_fields,
        n_spatial_dims=2,
        spatial_resolution=tuple(cfg.data.target_hw),
        init_features=cfg.model.init_features,
        emb_dim=cfg.model.emb_dim,
        cond_channels=cfg.model.cond_channels,
        n_steps_output=cfg.data.n_steps_output,
    )

    summary(model, depth=5)

    # -----------------------------------------------------------------
    # DEVICE
    # -----------------------------------------------------------------
    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
        model = model.to(device)
        # NOTE: You can wrap with DDP if you want later
        # model = DDP(model, device_ids=[local_rank])
    else:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model = model.to(device)

    logger.info(f"Model device: {device}")
    logger.info(f"device = {device}")


    # -----------------------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------------------
    optimizer = None
    if not validation_mode:
        logger.info("Instantiate optimizer torch.optim.AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

    # -----------------------------------------------------------------
    # LR SCHEDULER
    # -----------------------------------------------------------------
    if hasattr(cfg, "lr_scheduler") and not validation_mode:
        logger.info("Instantiate LR scheduler LinearWarmupCosineAnnealingLR")
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg.lr_scheduler.warmup_epochs,
            max_epochs=cfg.trainer.epochs,
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None

    # -----------------------------------------------------------------
    # TRAINER
    # -----------------------------------------------------------------
    logger.info("Instantiate Trainer")

    print("--------------------------------------------------", datamodule)
    trainer = Trainer(
        checkpoint_folder=checkpoint_folder,
        artifact_folder=artifact_folder,
        viz_folder=viz_folder,
        formatter="channels_first_default",
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        loss_fn=MSE(),
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

    trainer.show_progress = False

    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------
    if validation_mode:
        trainer.validate()
    else:
        # Save config
        with open(osp.join(experiment_folder, "extended_config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

        trainer.train()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    # Torch optimization settings (same as train.py)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # If user doesn't have configs/multi_pde.yaml, fall back to flat multi_dataset.yaml
    # This makes it usable immediately.
    if cfg is None or len(cfg.keys()) == 0:
        logger.warning(
            f"No Hydra config loaded. Falling back to {FALLBACK_FLAT_DATA_CONFIG}"
        )
        cfg = OmegaConf.load(str(FALLBACK_FLAT_DATA_CONFIG))

    cfg = _ensure_nested_cfg(cfg)

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

    # -----------------------------------------------------------------
    # WANDB
    # -----------------------------------------------------------------
    wandb_group = f"multi_{'_'.join(cfg.data.train_datasets[:2])}"
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

    # DDP disabled like train.py currently
    is_distributed, world_size, rank, local_rank = False, 1, 0, 0

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
