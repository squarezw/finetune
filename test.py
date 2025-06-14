import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

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
    prompt = (
        "根据以下房屋特征预测房价:\n"
        f"- 经度: {features_dict['经度']}\n"
        f"- 纬度: {features_dict['纬度']}\n"
        f"- 房龄中位数: {features_dict['房龄中位数']}年\n"
        f"- 总房间数: {features_dict['总房间数']}\n"
        f"- 总卧室数: {features_dict['总卧室数']}\n"
        f"- 人口: {features_dict['人口']}\n"
        f"- 家庭数: {features_dict['家庭数']}\n"
        f"- 收入中位数: {features_dict['收入中位数']}\n"
        "预测房价(美元):"
    )
    print("prompt：", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=32)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("模型原始输出：", prediction)
    # 提取所有数字，取最后一个
    numbers = re.findall(r"([0-9]+)", prediction)
    if numbers:
        return float(numbers[-1])
    else:
        print(f"解析失败: {prediction}")
        return None
    
# 测试示例
test_case = {
    "经度": -118.24,
    "纬度": 34.05,
    "房龄中位数": 25.0,
    "总房间数": 2000.0,
    "总卧室数": 400.0,
    "人口": 1200.0,
    "家庭数": 350.0,
    "收入中位数": 4.5
}

print(f"预测结果: ${predict_house_price(test_case):,.0f}")