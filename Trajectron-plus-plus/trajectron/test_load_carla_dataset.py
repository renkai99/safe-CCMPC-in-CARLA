

import sys

sys.path.append('/home/cchen795/scratch/src/carla-0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
sys.path.append('/home/cchen795/scratch/code/python-utility/utility')
sys.path.append('/home/cchen795/scratch/code/python-utility/carlautil')
sys.path.append('/home/cchen795/scratch/code/trajectron-plus-plus/experiments/nuScenes')
sys.path.append('/home/cchen795/scratch/code/carla-collect')

import dill
with open('/home/cchen795/scratch/nuScenes/processed/carla_val_v1_full.pkl', 'rb') as f:
    env = dill.load(f, encoding='latin1')
print( len(env.scenes) )

with open('/home/cchen795/scratch/nuScenes/processed/carla_test_v1_full.pkl', 'rb') as f:
    env = dill.load(f, encoding='latin1')
print( len(env.scenes) )

with open('/home/cchen795/scratch/nuScenes/processed/carla_train_v1_full.pkl', 'rb') as f:
    env = dill.load(f, encoding='latin1')
print( len(env.scenes) )

