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


# Step 1: Update Amazon Linux 2
sudo yum update -y

# Step 2: Install the amazon-Linux-extras package
sudo yum install -y amazon-linux-extras

# Step 3: Enable EPEL Repository
sudo amazon-linux-extras install epel -y
sudo yum update -y

# Step 4: Installing git LFS in Amazon Linux 2
sudo yum install -y git-lfs

# Step 5: Verify Installation
echo "Git LFS installed successfully."


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
