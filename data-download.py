from datasets import load_dataset

# 加载数据集
dataset = load_dataset('openai/gsm8k', 'main')

# 保存训练集到本地
dataset['train'].to_parquet('sample_data/gsm8k_train.parquet')
dataset['test'].to_parquet('sample_data/gsm8k_test.parquet')

# 或保存为 CSV 格式
dataset['train'].to_csv('sample_data/gsm8k_train.csv')
dataset['test'].to_csv('sample_data/gsm8k_test.csv')