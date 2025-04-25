#!/bin/bash

# ========== GPU Selection ==========
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# ========== Project Setup ==========
NAME=crofts_lot_gsplats
PROJECT=CityGSV2_CroftsLot
CONFIG_PATH=configs/crofts_lot.yaml
DATA_PATH=/vol/data/Crofts_Lot

# ========== Step 1: Train Model ==========
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
    --config $CONFIG_PATH \
    -n $NAME \
    --logger wandb \
    --project $PROJECT \
    --data.path $DATA_PATH

# ========== Step 2: Evaluate and Save Outputs ==========
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$NAME/config.yaml \
    --data.path $DATA_PATH \
    --data.parser.eval_image_select_mode ratio \
    --data.parser.eval_ratio 1.0 \
    --save_val

# ========== Step 3: Extract Mesh from Splats ==========
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python utils/gs2d_mesh_extraction.py \
    outputs/$NAME \
    --voxel_size 0.01 \
    --sdf_trunc 0.04 \
    --depth_trunc 5.0

echo "✅ All steps completed for $NAME!"
