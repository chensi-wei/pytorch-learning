import torch
import torch.nn as nn
import torch.optim as optim

# ======================
# 1. 数据
# ======================
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0]
])
y_true = torch.tensor([[6.0], [9.0], [12.0]])

# ======================
# 2. 模型
# ======================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # 输入3个特征 → 隐藏层5个神经元
        self.fc2 = nn.Linear(5, 1)  # 隐藏层5个 → 输出1个结果

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU激活函数，让模型能学习复杂关系
        x = self.fc2(x)
        return x

model = SimpleNet()

# ======================
# 3. 损失 & 优化器
# ======================
criterion = nn.MSELoss()  # 均方误差损失，适合回归任务（预测具体数值）
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器，自动改参数

# ======================
# 4. 标准训练循环
# ======================
print("开始训练...")
for epoch in range(300):
    # 前向传播：让模型猜答案
    outputs = model(x)
    # 计算损失：模型猜的和真实答案差多少
    loss = criterion(outputs, y_true)

    # 反向传播 + 优化（核心三步，实习面试必问！）
    optimizer.zero_grad()  # 清空上一轮的梯度，避免叠加错误
    loss.backward()        # 自动计算梯度（找错在哪）
    optimizer.step()       # 自动更新模型参数（改错）

    # 打印日志，看损失是否下降（损失越小，模型越准）
    if epoch % 50 == 0:
        print(f"Epoch: {epoch:3d} | Loss: {loss.item():.4f}")

# ======================
# 5. 测试模型
# ======================
model.eval()  # 切换到测试模式，关闭梯度计算（节省资源）
with torch.no_grad():  # 禁止计算梯度，避免占用内存
    predictions = model(x)

print("\n最终预测结果:")
print(predictions)

# ----------------------
# 我的个人理解（学习笔记）
# ----------------------
# 1. 整个训练流程就是：猜答案→算误差→找错因→改参数，重复300次
# 2. 自定义模型必须继承nn.Module，重写__init__（搭层）和forward（前向传播）
# 3. ReLU激活函数的作用：让模型不再是简单的线性计算，能学习更复杂的规律
# 4. optimizer.zero_grad() 必须写在loss.backward()前面，否则梯度会叠加，模型学歪
# 5. model.eval() 和 with torch.no_grad() 是测试的固定写法，避免多余计算
# 6. 训练后loss会逐渐下降，说明模型在不断优化，最终预测结果接近真实答案
# 7. 这是PyTorch实战的核心模板，不管是回归还是分类，都能套用这个框架
