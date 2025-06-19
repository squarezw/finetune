FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# 安装 Python3 venv 和 pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 拷贝依赖文件
COPY requirements.txt .

# 创建虚拟环境并安装依赖
RUN python3 -m venv /workspace/venv && \
    /workspace/venv/bin/pip install --upgrade pip && \
    /workspace/venv/bin/pip install -r requirements.txt

# 拷贝代码
COPY . .

# 默认用 venv 的 python 运行
ENV PATH="/workspace/venv/bin:$PATH"
