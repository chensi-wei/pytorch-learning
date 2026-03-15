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
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()

# ======================
# 3. 损失 & 优化器
# ======================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ======================
# 4. 标准训练循环
# ======================
print("开始训练...")
for epoch in range(300):
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y_true)

    # 反向传播 & 优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印日志
    if epoch % 50 == 0:
        print(f"Epoch: {epoch:3d} | Loss: {loss.item():.4f}")

# ======================
# 5. 测试模型
# ======================
model.eval()
with torch.no_grad():
    predictions = model(x)

print("\n最终预测结果:")
print(predictions)
