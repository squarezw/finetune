import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 基础模型和 LoRA adapter 路径
base_model_name = "Qwen/Qwen1.5-0.5B"
lora_model_path = "./qwen3-7b-finetuned"

# 加载基础模型和 LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, lora_model_path)
tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

# # 测试微调后的模型
# def generate_response(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     outputs = model.generate(**inputs, max_new_tokens=200)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # 测试示例
# test_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain what artificial intelligence is.\n\n### Response:"
# print(generate_response(test_prompt))


def predict_house_price(features_dict):
    # 准备输入
    prompt = "房屋特征: " + " ".join(
        f"{k}:{v}" for k, v in features_dict.items()
    ) + " 预测房价:"
    
    # 生成预测
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=8)
    
    # 解析输出
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_value = prediction.split("预测房价:")[-1].strip()
    
    try:
        return float(predicted_value)
    except:
        print(f"解析失败: {predicted_value}")
        return None
    
# 测试示例
test_case = {
    "longitude": -118.24,
    "latitude": 34.05,
    "housing_median_age": 25.0,
    "total_rooms": 2000.0,
    "total_bedrooms": 400.0,
    "population": 1200.0,
    "households": 350.0,
    "median_income": 4.5
}
print(f"预测结果: ${predict_house_price(test_case):,.0f}")