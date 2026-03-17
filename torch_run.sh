export CUDA_VISIBLE_DEVICES=0

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/mnt/data/cyx/Engram-Planner/df/bin/python"

TRAIN_SET_PATH="/mnt/data/cyx/dp_training_data_mini"
TRAIN_SET_LIST_PATH="/mnt/data/cyx/Engram-Planner/diffusion_planner_training.json"
###################################

sudo -E $RUN_PYTHON_PATH -m torch.distributed.run \
  --nnodes 1 \
  --nproc-per-node 1 \
  --standalone \
  --master_port 29517 \
  train_predictor.py \
  --train_set $TRAIN_SET_PATH \
  --train_set_list $TRAIN_SET_LIST_PATH \
  --resume_model_path /mnt/data/cyx/Engram-Planner/training_log/df-engram/2026-03-11-15:05:27 \
  --batch_size 256