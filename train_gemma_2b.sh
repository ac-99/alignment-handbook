#!/bin/bash

# This script sets up the environment and runs the main script.

# Setup Environment

# Create and activate virtual environment
python3 -m venv gemma-2b-fine-tune
source gemma-2b-fine-tune/bin/activate


# Install specific version of Torch
pip3 install torch==2.1.2

# Install the Alignment Handbook package
python -m pip install .

pip install wheel

# Install Flash-Attn library without build isolation
python -m pip install flash-attn==2.3.6 --no-build-isolation

# Login to the Hugging Face Hub
python -c "from huggingface_hub import login; login()"


yum install -y amazon-linux-extras

# Install epel-release from Extras Library
echo "Installing epel-release..."
amazon-linux-extras install epel -y 

# Enable epel repository
echo "Enabling epel repository..."
yum-config-manager --enable epel

# Install git-lfs
echo "Installing git-lfs..."
yum install git-lfs -y

echo "git-lfs installation completed."

# Install Weights & Biases library
pip install wandb

# Install specific version of Torch
pip3 install torch==2.1.2
pip install cuda==11.0

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export WANDB_DISABLED="false"

# Run the Main Script

# Set log level for Accelerate library
ACCELERATE_LOG_LEVEL=info

# Launch training with specific configurations and scripts using Accelerate
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-2b-gemma/sft/config_full.yaml

accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-2b-gemma/dpo/config_full.yaml
