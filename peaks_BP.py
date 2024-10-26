import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 定义 Peaks 函数
def peaks(x, y):
    return 3 * (1 - x) ** 2 * torch.exp(-(x ** 2) - (y + 1) ** 2) - \
        10 * (x / 5 - x ** 3 - y ** 5) * torch.exp(-x ** 2 - y ** 2) - \
        1 / 3 * torch.exp(-(x + 1) ** 2 - y ** 2)


# 生成训练数据
def generate_data(num_samples, sample_type='random'):
    if sample_type == 'random':
        x_train = torch.FloatTensor(num_samples).uniform_(-4, 4)
        y_train = torch.FloatTensor(num_samples).uniform_(-4, 4)
    elif sample_type == 'feature_based':
        half_samples = num_samples // 2
        x_train_random = torch.FloatTensor(half_samples).uniform_(-4, 4)
        y_train_random = torch.FloatTensor(half_samples).uniform_(-4, 4)

        # 特征区域采样（接近函数极值区域，例如 [-2, 2] 内的区域）
        x_train_feature = torch.FloatTensor(half_samples).uniform_(-2, 2)
        y_train_feature = torch.FloatTensor(half_samples).uniform_(-2, 2)

        x_train = torch.cat([x_train_random, x_train_feature], dim=0)
        y_train = torch.cat([y_train_random, y_train_feature], dim=0)

    z_train = peaks(x_train, y_train)
    return x_train, y_train, z_train


# 定义BP网络
class BPNetwork(nn.Module):
    def __init__(self):
        super(BPNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 7)
        self.fc2 = nn.Linear(7, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练模型
def train_model(x_train, y_train, z_train, learning_rate=0.1, momentum=0.4, steps=15000):
    model = BPNetwork()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    inputs = torch.stack([x_train, y_train], dim=1)
    targets = z_train.view(-1, 1)

    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if step % 3000 == 0:
            print(f'Step {step}, Loss: {loss.item():.4f}')
    return model


# 可视化结果
def plot_results(model, num_samples, sample_type):
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x, y)
    inputs = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        Z_pred = model(inputs).view(100, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.numpy(), Y.numpy(), Z_pred.numpy(), cmap='viridis')
    plt.title(f'Approximation with {num_samples} samples ({sample_type} sampling)')
    plt.show()


# 主程序
sample_configs = [
    (150, 'random'),
    (200, 'feature_based')
]

for num_samples, sample_type in sample_configs:
    x_train, y_train, z_train = generate_data(num_samples, sample_type)
    model = train_model(x_train, y_train, z_train)
    plot_results(model, num_samples, sample_type)
