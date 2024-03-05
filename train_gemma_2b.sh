#!/bin/bash

# This script sets up the environment and runs the main script.

# Setup Environment

# Install specific version of Torch
pip3 install torch==2.1.2

# Install the Alignment Handbook package
python -m pip install .

# Install Flash-Attn library without build isolation
python -m pip install flash-attn==2.3.6 --no-build-isolation

# Login to the Hugging Face Hub
python -c "from huggingface_hub import login; login()"

# Install git-lfs for managing large files
yum install git-lfs

# Install Weights & Biases library
pip install wandb

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export WANDB_DISABLED="false"

# Run the Main Script

# Set log level for Accelerate library
ACCELERATE_LOG_LEVEL=info

# Launch training with specific configurations and scripts using Accelerate
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-2b-gemma/sft/config_full.yaml

accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-2b-gemma/dpo/config_full.yaml
