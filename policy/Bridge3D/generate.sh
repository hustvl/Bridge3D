#!/bin/bash

# Resolve project-relative paths from this script's directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

model_name=${1}

python ./model_config/_generate_model_config.py $model_name 
