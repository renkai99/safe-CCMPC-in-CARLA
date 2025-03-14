#!/bin/bash

source ../../../env.sh

DATA_VER=v3-1-1
DATA_DIR=$HOME/code/robotics/carla-collect/carla_${DATA_VER}_dataset
DATA_PATH=$DATA_DIR/${DATA_VER}_split1_test.pkl

DATE_DIR=20210801
MODEL_NAME=models_01_Aug_2021_20_02_57_carla_v3-1-1_base_distmapV3_ph8
TAG_NAME=models_01_Aug_2021_20_02_57_carla_v3-1-1_base_distmapV3

python evaluate.py \
	--model models/$DATE_DIR/$MODEL_NAME \
	--data $DATA_PATH \
	--checkpoint=20 \
	--output_path results \
	--output_tag $TAG_NAME \
	--node_type VEHICLE \
	--prediction_horizon 2

python evaluate.py \
	--model models/$DATE_DIR/$MODEL_NAME \
	--data $DATA_PATH \
	--checkpoint=20 \
	--output_path results \
	--output_tag $TAG_NAME \
	--node_type VEHICLE \
	--prediction_horizon 4

python evaluate.py \
	--model models/$DATE_DIR/$MODEL_NAME \
	--data $DATA_PATH \
	--checkpoint=20 \
	--output_path results \
	--output_tag $TAG_NAME \
	--node_type VEHICLE \
	--prediction_horizon 6

python evaluate.py \
	--model models/$DATE_DIR/$MODEL_NAME \
	--data $DATA_PATH \
	--checkpoint=20 \
	--output_path results \
	--output_tag $TAG_NAME \
	--node_type VEHICLE \
	--prediction_horizon 8
