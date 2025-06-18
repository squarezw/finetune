docker run -itd --name ml-env --gpus all -v "$(pwd)":/workspace -w /workspace pytorch/pytorch:latest bash

# 容器内
# apt-get update
# apt-get install -y gcc g++
# source /opt/conda/etc/profile.d/conda.sh
# conda create -n myenv python=3.12
# conda activate myenv
# pip install -r requirements.txt
# pip install -e .