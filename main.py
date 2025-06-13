import torch
print(f"GPU 可用: {torch.cuda.is_available()}")
print(f"GPU 型号: {torch.cuda.get_device_name(0)}")