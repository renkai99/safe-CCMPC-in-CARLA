#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=train_dynmap_ph8
#SBATCH --output=/scratch/cchen795/slurm/%x-%j.out
#SBATCH --error=/scratch/cchen795/slurm/%x-%j.out

# echo "load modules"
# module load python/3.6
# module load ipython-kernel/3.6
# module load geos

echo "set environment"
source  ../../py37.env.sh

DATA_DIR=../experiments/processed

echo "train"
export MPLBACKEND="agg"
python train.py \
    --seed $RANDOM \
    --data_dir $DATA_DIR \
    --eval_every 1 \
    --vis_every 1 \
    --conf ./int_ee_me.ph8.config.json \
    --train_data_dict nuScenes_train_full.pkl \
    --eval_data_dict nuScenes_val_full.pkl \
    --offline_scene_graph yes \
    --preprocess_workers 10 \
    --batch_size 256 \
    --log_dir ../experiments/nuScenes/models \
    --train_epochs 20 \
    --node_freq_mult_train \
    --log_tag _int_ee_me_ph8 \
    --map_encoding \
    --augment \
