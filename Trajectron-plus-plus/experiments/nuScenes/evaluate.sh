#!/bin/bash

conda activate trajectron

export PYTHONPATH=$HOME/src/carla-0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:$PYTHONPATH
export PYTHONPATH=$HOME/code/robotics/carla-collect:$PYTHONPATH
export PYTHONPATH=$HOME/code/robotics/trajectron-plus-plus/experiments/nuScenes:$PYTHONPATH

DATA_VER=v3_0_1
DATA_DIR=$HOME/code/robotics/carla-collect/carla_${DATA_VER}_dataset
DATA_PATH=$DATA_DIR/carla_test_${DATA_VER}_full.pkl

MODEL_NAME=models_20_Jul_2021_11_48_11_carla_v3_0_1_base_distmap_ph8
TAG_NAME=models_20_Jul_2021_11_48_11_carla_v3_0_1_base_distmap
python evaluate.py \
	--model models/$MODEL_NAME \
	--data $DATA_PATH \
	--checkpoint=20 \
	--output_path results \
	--output_tag $TAG_NAME \
	--node_type VEHICLE \
	--prediction_horizon 8
