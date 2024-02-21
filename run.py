
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL) # Filter warnings for less verbosity
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pdb

from pytorch_lightning import seed_everything
seed_everything(42)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize datamodule
    dm = hydra.utils.instantiate(cfg.datamodule, \
        branch = cfg.branch,
        temp_root = cfg.temp_dataset.root,
        spat_root = cfg.spat_dataset.root,
        temp_img_file_template = cfg.temp_dataset.img_file_template,
        num_segments = cfg.temp_dataset.num_segments,
        use_sifar = cfg.temp_dataset.use_sifar,
        temp_augs_enable_center_crop= cfg.temp_dataset.temp_augs_enable_center_crop,
        temp_augs_enable_multiscale_jitter= cfg.temp_dataset.temp_augs_enable_multiscale_jitter,
        temp_augs_enable_mixup= cfg.temp_dataset.temp_augs_enable_mixup,
        temp_augs_enable_cutmix= cfg.temp_dataset.temp_augs_enable_cutmix,
        temp_augs_enable_erasing= cfg.temp_dataset.temp_augs_enable_erasing,
        temp_augs_enable_augmix= cfg.temp_dataset.temp_augs_enable_augmix,
        temp_augs_enable_normalize= cfg.temp_dataset.temp_augs_enable_normalize,
        temp_augs_use_sifar= cfg.temp_dataset.temp_augs_use_sifar,
    )
    dm.setup()

    aug_str = ""

    if cfg.temp_dataset.temp_augs_enable_center_crop:
        aug_str += "_centercrop"
    if cfg.temp_dataset.temp_augs_enable_multiscale_jitter:
        aug_str += "_multiscalejitter"
    if cfg.temp_dataset.temp_augs_enable_mixup:
        aug_str += "_mixup"
    if cfg.temp_dataset.temp_augs_enable_cutmix:
        aug_str += "_cutmix"
    if cfg.temp_dataset.temp_augs_enable_erasing:
        aug_str += "_erasing"
    if cfg.temp_dataset.temp_augs_enable_augmix:
        aug_str += "_augmix"
    if cfg.temp_dataset.temp_augs_enable_normalize:
        aug_str += "_normalize"

    run_name = cfg.wandb.name + aug_str
    print("Run name: ", run_name)

    # Initialize wandb logger
    cfg.wandb.name = run_name
    wandb_logger = hydra.utils.instantiate(cfg.wandb) if cfg.logging else None

    # Initialize model
    model = None
    if cfg.branch == 'temporal':
        model = hydra.utils.instantiate(cfg.temporal, \
            num_subjects=cfg.temp_dataset.num_subjects, 
            num_verbs=cfg.temp_dataset.num_verbs, 
            num_targets=cfg.temp_dataset.num_targets,
            run_name=cfg.wandb.name,
            log_cfms=cfg.log_cfms,
        )
    elif cfg.branch == 'spatial':
        model = hydra.utils.instantiate(cfg.spatial, \
            n_channels=cfg.spat_dataset.n_channels, 
            n_classes=cfg.spat_dataset.n_classes, 
        )
    
    # Initialize callbacks
    es_callback = hydra.utils.instantiate(cfg.es_callback)
    ckpt_callback = hydra.utils.instantiate(
        cfg.ckpt_callback, 
        filename=str(cfg.wandb.name)+"_"+"lapnet-{epoch:02d}-{val_loss:.3f}-{val_acc_subject:.3f}-{val_acc_verb:.3f}-{val_acc_target:.3f}-{val_mean_acc:.3f}"
    )

    # Initialize trainer
    ddstrat = hydra.utils.instantiate(cfg.ddstrat)
    trainer = hydra.utils.instantiate(cfg.trainer, strategy=ddstrat, logger=wandb_logger, callbacks=[es_callback, ckpt_callback])
    
    # Train model
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # Inference on test set
    trainer.test(model, dm.val_dataloader(), ckpt_path='best')


if __name__ == '__main__':
    main()