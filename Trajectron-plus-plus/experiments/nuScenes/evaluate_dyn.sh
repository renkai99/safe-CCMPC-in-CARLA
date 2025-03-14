
#!/bin/bash

python evaluate.py \
    --model models/int_ee \
    --checkpoint=12 \
    --data ../processed/nuScenes_test_full.pkl \
    --output_path results \
    --output_tag int_ee \
    --node_type VEHICLE \
    --prediction_horizon 6
