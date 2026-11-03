#!/bin/bash

# Resolve project-relative paths from this script's directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

task_name=click_bell
task_config=demo_clean

expert_data_num=50
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}
python scripts/process_data_bridge3d.py $task_name $task_config $expert_data_num
