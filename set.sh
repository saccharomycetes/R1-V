conda activate r1v

# Install CUDA 12.6 toolkit (matches common PyTorch 12.6 wheels; includes nvcc)
conda install -y -c nvidia cuda-toolkit=12.6

# Point build tools to this CUDA (avoid picking up a system CUDA)
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# (Optional but recommended) set archs for FA build; adjust to your GPUs
# A100=8.0, L40S/ADA=8.9, H100/H200=9.0
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"

# Sanity checks
nvcc -V
python -c "import torch, sys; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"