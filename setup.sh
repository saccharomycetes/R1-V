# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
pip install wandb
pip install tensorboardx
pip install qwen_vl_utils torchvision

# Ensure correct torch version
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Flash attention - specify version
pip install flash-attn==2.8.0.post2 --no-build-isolation

# vLLM support - specify version
pip install vllm==0.10.1.1

# fix transformers version
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef