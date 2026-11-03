#!/bin/bash

# Resolve project-relative paths from this script's directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

task_name=${1}
task_config=${2}
expert_data_num=${3}
gpu_id=${4}

export CUDA_VISIBLE_DEVICES=${gpu_id}
python scripts/process_data.py $task_name $task_config $expert_data_num
