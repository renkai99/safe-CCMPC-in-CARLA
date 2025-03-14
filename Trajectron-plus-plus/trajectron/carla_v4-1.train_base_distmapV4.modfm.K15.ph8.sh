#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=carla_v4-1.train_base_distmapV4.modfm.K15.ph8
#SBATCH --output=/scratch/cchen795/1/slurm/%x-%j.out
#SBATCH --error=/scratch/cchen795/1/slurm/%x-%j.out

echo "load modules"
module load python/3.8
module load ipython-kernel/3.8
module load geos

echo "activate virtualenv"
source $HOME/scratch/py38torch104trajectron/bin/activate

echo "environmental variables"
source  ../../py38cc.env.sh

DATA_VER=v4-1
DATA_DIR=../experiments/processed

echo "train"
export MPLBACKEND="agg"
python train.py \
	--seed $RANDOM \
	--data_dir $DATA_DIR \
	--eval_every 1 \
	--vis_every 1 \
	--conf base_distmapV4-1.K15.ph8.json \
	--train_data_dict ${DATA_VER}_split1_train_modfm.pkl \
	--eval_data_dict ${DATA_VER}_split1_val.pkl \
	--offline_scene_graph yes \
	--preprocess_workers 10 \
	--batch_size 256 \
	--log_dir ../experiments/nuScenes/models \
	--train_epochs 20 \
	--node_freq_mult_train \
	--log_tag _carla_${DATA_VER}_base_distmapV4_modfm_K15_ph8 \
	--map_encoding \
	--augment \

