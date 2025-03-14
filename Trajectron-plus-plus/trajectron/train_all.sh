#!/bin/bash

#	--train_data_dict nuScenes_train_mini_full.pkl \
#	--eval_data_dict nuScenes_test_mini_full.pkl \
python train.py \
	--eval_every 1 \
	--vis_every 1 \
	--conf ../experiments/nuScenes/models/robot/config.json \
	--train_data_dict nuScenes_train_full.pkl \
	--eval_data_dict nuScenes_test_full.pkl \
	--offline_scene_graph yes \
	--preprocess_workers 10 \
	--batch_size 256 \
	--log_dir ../experiments/nuScenes/models \
	--train_epochs 20 \
	--node_freq_mult_train \
	--log_tag _robot \
	--incl_robot_node \
	--map_encoding
