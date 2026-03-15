import torch
import torch.nn as nn

# 最简单线性模型
model = nn.Linear(2, 1)

# 输入
x = torch.tensor([[1.0, 2.0]])

# 预测
y_pred = model(x)

print("输入:", x)
print("模型预测:", y_pred)
