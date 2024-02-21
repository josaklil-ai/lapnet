#!/bin/bash

#SBATCH --job-name=lapnet
#SBATCH --output=/dev/null
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT

source activate surg
python run.py logging=True \
    datamodule.temp_batch_size=16 \
    temp_dataset.temp_augs_enable_center_crop=True \
    temp_dataset.temp_augs_enable_multiscale_jitter=True \
    temp_dataset.temp_augs_enable_erasing=True 
