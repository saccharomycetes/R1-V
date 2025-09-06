cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export WANDB_API_KEY="b8d0eaae3a3e473649dea6984305a0a412740cb5"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir ./output \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name leonardPKU/clevr_cogen_a_train \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8
