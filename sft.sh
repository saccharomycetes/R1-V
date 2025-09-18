#!/bin/bash

# ========================================
# R1-V Supervised Fine-Tuning (SFT) Script
# ========================================

# ========================================
# Configuration Paths
# ========================================
export WANDB_API_KEY="b8d0eaae3a3e473649dea6984305a0a412740cb5"
ACCELERATE_CONFIG="src/r1-v/configs/zero2.yaml"
SFT_SCRIPT="src/r1-v/src/open_r1/sft.py"
SFT_CONFIG="src/r1-v/configs/qwen2vl_sft_config.yaml"

# ========================================
# Training Execution
# ========================================
accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    "$SFT_SCRIPT" \
    --config "$SFT_CONFIG"