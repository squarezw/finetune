import torch
print(f"GPU 可用: {torch.cuda.is_available()}")
print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

import bitsandbytes
import transformers
import pandas as pd
from datasets import Dataset


print(f"bitsandbytes 版本: {bitsandbytes.__version__}")
print(f"transformers 版本: {transformers.__version__}")

# 加载模型和分词器
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen1.5-7B"  # Qwen3 7B 模型

# 配置 4-bit 量化以减少内存使用
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)


# 加载CSV文件
try:
    df = pd.read_csv('sample_data/california_housing_train.csv')
    dataset = Dataset.from_pandas(df)
    print("CSV文件加载成功！")
    print(f"数据集结构: {dataset}")
    print("\n首条样本:", dataset[0])
    print("\n所有列名:", dataset.column_names)

except Exception as e:
    print(f"加载失败: {e}")
    
    # 创建测试数据作为后备
    test_data = pd.DataFrame({
        'pixel0': [0, 128],
        'label': [0, 1]
    })
    dataset = Dataset.from_pandas(test_data)
    print("使用测试数据继续运行")