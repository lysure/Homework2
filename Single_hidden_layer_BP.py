import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib import font_manager



# 原始样本数
X = np.array([[0.75, 1.0], [0.5, 0.75], [0.25, 0.0],  # 类别1
              [0.5, 0.0], [0.0, 0.0], [1.0, 0.75],  # 类别2
              [1.0, 1.0], [0.5, 0.25], [0.75, 0.5]])  # 类别3
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # 三类标签

# 增加噪声
delta = 0.25
noise = np.random.uniform(-delta, delta, X.shape)
X_noisy = X + noise

# 数据归一化到区间 [-1, 1]
X_min, X_max = X_noisy.min(), X_noisy.max()
X_norm = 2 * (X_noisy - X_min) / (X_max - X_min) - 1

# 定义并训练单隐层BP网络
hidden_layer_sizes = [2, 3, 4, 5]  # 尝试不同的隐层节点个数
for nodes in hidden_layer_sizes:
    bp_model = MLPClassifier(hidden_layer_sizes=(nodes,), max_iter=10000, activation='tanh', random_state=1)
    bp_model.fit(X_norm, y)

    # 生成分类边界可视化
    x_min, x_max = X_norm[:, 0].min() - 0.5, X_norm[:, 0].max() + 0.5
    y_min, y_max = X_norm[:, 1].min() - 0.5, X_norm[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = bp_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(f'BP神经网络分类结果 (隐层节点数: {nodes})')
    plt.xlabel('归一化后的 x1')
    plt.ylabel('归一化后的 x2')
    plt.show()
