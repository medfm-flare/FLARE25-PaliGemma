#!/bin/bash

python inference.py \
    --base_dataset_path /workspace/inputs \
    --output_dir /workspace/outputs \
    --output_filename predictions.json \
    --max_new_tokens 1024 \
    --device cuda:0