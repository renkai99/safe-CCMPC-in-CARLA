conda activate trajectron-cplex
source  ../../env.sh

DATA_VER=v3-1
DATA_DIR=$APPROOT/carla_${DATA_VER}_dataset

python train.py \
	--seed $RANDOM \
	--data_dir $DATA_DIR \
	--eval_every 1 \
	--vis_every 1 \
	--conf base_distmap.ph8.json \
	--train_data_dict ${DATA_VER}_split1_train.pkl \
	--eval_data_dict ${DATA_VER}_split1_val.pkl \
	--offline_scene_graph yes \
	--preprocess_workers 3 \
	--batch_size 256 \
	--log_dir ../experiments/nuScenes/models \
	--train_epochs 20 \
	--node_freq_mult_train \
	--log_tag _carla_${DATA_VER}_base_distmap_ph8 \
	--map_encoding \
	--augment \

