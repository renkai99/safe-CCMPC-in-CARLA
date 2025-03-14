conda activate trajectron

export PYTHONPATH=$HOME/src/carla-0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:$PYTHONPATH
export PYTHONPATH=$HOME/code/robotics/carla-collect:$PYTHONPATH
export PYTHONPATH=$HOME/code/robotics/trajectron-plus-plus/experiments/nuScenes:$PYTHONPATH

DATA_VER=v3_0_1
DATA_DIR=$HOME/code/robotics/carla-collect/carla_${DATA_VER}_dataset

python train.py \
	--seed $RANDOM \
	--data_dir $DATA_DIR \
	--eval_every 1 \
	--vis_every 1 \
	--conf base_distmap.json \
	--train_data_dict carla_train_${DATA_VER}_full.pkl \
	--eval_data_dict carla_val_${DATA_VER}_full.pkl \
	--offline_scene_graph yes \
	--preprocess_workers 5 \
	--batch_size 256 \
	--log_dir ../experiments/nuScenes/models \
	--train_epochs 20 \
	--node_freq_mult_train \
	--log_tag _carla_${DATA_VER}_base_distmap_ph6 \
	--map_encoding \
	--augment \

