
# Instructions to Replicate Zephyr 2B Gemma

Similar to how Zephyr 7B Gemma was trained, training this model proceeds in two steps:

1. Apply SFT to fine-tune Gemma 2B on the Deita 10k dataset.
2. Align the SFT model to AI feedback via DPO on a curated mix of 7k examples by Argilla ([link](https://huggingface.co/datasets/argilla/dpo-mix-7k)). 

See below for commands to train these models using either DeepSpeed ZeRO-3 or LoRA.

## Full training examples

You will require 8 GPUs (80GB of VRAM) to train the full model - alternatively, you can train on 1 GPU by adjusting the micro batch size and gradient accumulation steps to keep the global batch size constant. A recipe involving QLoRA will come later 🤗.

```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-2b-gemma/sft/config_full.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-2b-gemma/dpo/config_full.yaml
```
