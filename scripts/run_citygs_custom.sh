#!/bin/bash

# ========== Usage Check ==========
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <DATA_PATH> [NAME] [PROJECT_NAME]"
  echo "Example: $0 /vol/data/Crofts_Lot crofts_lot_gsplats CityGSV2_CroftsLot"
  exit 1
fi

# ========== Input Arguments ==========
DATA_PATH="$1"
NAME="${2:-$(basename "$DATA_PATH")_gsplats}"  # default to foldername_gsplats
PROJECT="${3:-CityGSV2_$NAME}"                 # default to CityGSV2_foldername_gsplats
CONFIG_PATH="configs/custom_aerial.yaml"       # config stays fixed for now
DOWNSAMPLE_RATIO=1

# ========== GPU Selection ==========
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# ========== Step 0: Estimate Depth ==========
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available for depth estimation."
CUDA_VISIBLE_DEVICES=$gpu_id python utils/estimate_dataset_depths.py "$DATA_PATH" -d $DOWNSAMPLE_RATIO

# ========== Step 1: Train Model ==========
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available for training."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
    --config $CONFIG_PATH \
    -n "$NAME" \
    --project "$PROJECT" \
    --data.path "$DATA_PATH"

# ========== Step 2: Evaluate and Save Outputs ==========
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available for evaluation."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$NAME/config.yaml \
    --data.path "$DATA_PATH" \
    --data.parser.eval_image_select_mode ratio \
    --data.parser.eval_ratio 1.0 \
    --save_val

# ========== Step 3: Extract Mesh from Splats ==========
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available for mesh extraction."
CUDA_VISIBLE_DEVICES=$gpu_id python utils/gs2d_mesh_extraction.py \
    outputs/$NAME \
    --voxel_size 0.01 \
    --sdf_trunc 0.04 \
    --depth_trunc 5.0

echo "✅ All steps completed for $NAME!"
