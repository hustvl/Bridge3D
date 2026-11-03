#!/bin/bash

# Resolve project-relative paths from this script's directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
policy_name=Bridge3D
task_name='click_bell'  # [TODO]
task_config='demo_clean'
model_name='click_bell_demo_clean_bridge3d'
checkpoint_id=10000
seed=0
gpu_id=0

DEBUG=False
export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --checkpoint_id ${checkpoint_id} \
    --policy_name ${policy_name} 
