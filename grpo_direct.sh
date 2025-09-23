export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export WANDB_API_KEY="b8d0eaae3a3e473649dea6984305a0a412740cb5"

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/r1-v/src/open_r1/grpo.py \
    --output_dir ./output/grpo_direct \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name blocks/hard_data \
    --deepspeed src/r1-v/local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen25-VL-3B-GRPO-Blocks-Direct \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4 \
    --think false
