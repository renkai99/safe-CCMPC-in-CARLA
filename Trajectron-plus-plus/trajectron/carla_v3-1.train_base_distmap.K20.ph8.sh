#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=carla_v3-1.train_base_distmap.ph8
#SBATCH --output=/scratch/cchen795/slurm/%x-%j.out
#SBATCH --error=/scratch/cchen795/slurm/%x-%j.out

echo "load modules"
module load python/3.6
module load ipython-kernel/3.6
module load geos

echo "activate virtualenv"
source $HOME/pytrajectron/bin/activate

echo "environmental variables"
source  ../../env.sh

DATA_VER=v3-1
DATA_DIR=../experiments/processed
K_VER=K20

echo "train"
export MPLBACKEND="agg"
python train.py \
	--seed $RANDOM \
	--data_dir $DATA_DIR \
	--eval_every 1 \
	--vis_every 1 \
	--conf base_distmap.${K_VER}.ph8.json \
	--train_data_dict ${DATA_VER}_split1_train.pkl \
	--eval_data_dict ${DATA_VER}_split1_val.pkl \
	--offline_scene_graph yes \
	--preprocess_workers 10 \
	--batch_size 256 \
	--log_dir ../experiments/nuScenes/models \
	--train_epochs 20 \
	--node_freq_mult_train \
	--log_tag _carla_${DATA_VER}_base_distmap_${K_VER}_ph8 \
	--map_encoding \
	--augment \

