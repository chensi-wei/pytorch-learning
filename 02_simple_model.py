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

# ----------------------
# 我的个人理解（学习笔记）
# ----------------------
# 1. nn.Linear(2, 1) 表示：输入2个特征，输出1个结果（就是一个简单的线性计算器）
# 2. model = nn.Linear(2,1) 其实是创建了一个“线性层”，里面有weight（权重）和bias（偏置）
# 3. model(x) 就是让模型“前向传播”，本质是计算：x1*w1 + x2*w2 + b
# 4. 此时模型还没训练，所以预测结果是乱猜的，后续需要通过训练优化参数
