#!/bin/bash

#SBATCH --gres=gpu:1 -N 1 -n 6 -p smartbot

module load anaconda/2024.02
source activate robocasa

MUJOCO_GL=osmesa python robocasa/scripts/playback_dataset.py \
    --dataset /ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    --use-obs --render_image_names robot0_agentview_left

