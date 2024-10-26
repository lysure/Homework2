import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 定义 Peaks 函数
def peaks(x, y):
    return 3 * (1 - x) ** 2 * torch.exp(-(x ** 2) - (y + 1) ** 2) - \
        10 * (x / 5 - x ** 3 - y ** 5) * torch.exp(-x ** 2 - y ** 2) - \
        1 / 3 * torch.exp(-(x + 1) ** 2 - y ** 2)


# 生成训练数据，固定样本数量和采样方式
def generate_data(num_samples=200):
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


# 定义RBF网络，允许可调节的 beta 参数
class RBFNetwork(nn.Module):
    def __init__(self, num_centers, beta):
        super(RBFNetwork, self).__init__()
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.randn(num_centers, 2) * 8 - 4)
        self.beta = beta
        self.linear = nn.Linear(num_centers, 1, bias=False)

    def kernel_function(self, x, c):
        return torch.exp(-self.beta * torch.sum((x - c) ** 2, dim=1))

    def forward(self, x):
        phi = torch.stack([self.kernel_function(x, c) for c in self.centers], dim=1)
        return self.linear(phi)


# 训练模型
def train_model(x_train, y_train, z_train, num_centers, beta, learning_rate=0.01, steps=15000):
    model = RBFNetwork(num_centers=num_centers, beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
def plot_results(model, beta):
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    inputs = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        Z_pred = model(inputs).view(100, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.numpy(), Y.numpy(), Z_pred.numpy(), cmap='viridis')
    plt.title(f'RBF Approximation (beta={beta})')
    plt.show()


# 主程序，测试不同的 beta 值
betas = [0.1, 0.5, 1.0]  # 不同的尺度参数 beta 值
num_centers = 7  # 隐层神经元数量固定为7

# 固定样本数量和采样方式
x_train, y_train, z_train = generate_data()

# 对每个 beta 值训练模型并可视化结果
for beta in betas:
    model = train_model(x_train, y_train, z_train, num_centers=num_centers, beta=beta)
    plot_results(model, beta)
