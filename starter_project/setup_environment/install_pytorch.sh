echo "Installing PyTorch..."

# this is for PyTorch build 1.5.1 on Linux OS with CUDA Version 10.1"
# as of 07.10.2020, VM on GCP uses the Debian Linux distribution, which requires PyTorch v8.0 minimum
# can check Nvidia CUDA version with command "nvidia-smi"
# if your machine has different versions, can us this site to download the appropriate stable version: https://pytorch.org/get-started/locally/
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==0.8.1


echo "Installing APEX..."
# APEX is needed for mixed-precision training, e.g. floating point 16-bit option with Huggingface
# this tends to speed up training and reduce memory usage without losing performance (in some cases, model actually performs better due to improved generalization as a result of less precision in floats)
# this can take some time (usually ~15 min)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
