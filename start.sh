docker run -itd --name ml-env --gpus all -v "$(pwd)":/workspace -w /workspace pytorch/pytorch:latest bash
docker exec ml-env apt-get update
docker exec ml-env apt-get install -y gcc g++