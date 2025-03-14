#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_carla_v2_1_1_dyn_ph6
#SBATCH --output=/scratch/cchen795/slurm/%x-%j.out
#SBATCH --error=/scratch/cchen795/slurm/%x-%j.out

SCRATCH=/scratch/cchen795
echo "load modules"
module load python/3.6
module load ipython-kernel/3.6
module load geos

echo "activate virtualenv"
source $HOME/pytrajectron/bin/activate
APPDIR=$SCRATCH/code/trajectron-plus-plus/trajectron

export PYTHONPATH=/home/cchen795/scratch/src/carla-0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:/home/cchen795/scratch/code/python-utility/utility:/home/cchen795/scratch/code/python-utility/carlautil:/home/cchen795/scratch/code/trajectron-plus-plus/experiments/nuScenes:/home/cchen795/scratch/code/carla-collect:$PYTHONPATH

echo "train"
export MPLBACKEND="agg"
python train.py \
    --seed $RANDOM \
    --data_dir $SCRATCH/carla/processed \
    --eval_every 1 \
    --vis_every 1 \
    --conf ../experiments/nuScenes/models/int_ee/config.json \
    --train_data_dict carla_train_v2_1_1_full.pkl \
    --eval_data_dict carla_val_v2_1_1_full.pkl \
    --offline_scene_graph yes \
    --preprocess_workers 10 \
    --batch_size 256 \
    --log_dir ../experiments/nuScenes/models \
    --train_epochs 20 \
    --node_freq_mult_train \
    --log_tag _carla_v2_1_1_dyn_ph6 \
    --augment \

