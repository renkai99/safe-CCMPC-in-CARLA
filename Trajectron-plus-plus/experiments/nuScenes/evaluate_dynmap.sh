
#!/bin/bash

python evaluate.py \
    --model models/int_ee_me \
    --checkpoint=12 \
    --data ../processed/nuScenes_test_full.pkl \
    --output_path results \
    --output_tag int_ee_me \
    --node_type VEHICLE \
    --prediction_horizon 6
