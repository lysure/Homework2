import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder

# 训练数据
class1 = np.array([[0.75, 1.0], [0.5, 0.75], [0.25, 0.0]])
class2 = np.array([[0.5, 0.0], [0.0, 0.0], [1.0, 0.75]])
class3 = np.array([[1.0, 1.0], [0.5, 0.25], [0.75, 0.5]])

X = np.vstack((class1, class2, class3))
y = np.array([0] * 3 + [1] * 3 + [2] * 3).reshape(-1, 1)

# One-hot编码标签
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)


# 定义高斯径向基函数
def rbf(x, c, s):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * s ** 2))


# 构建RBF网络
class RBFNetwork:
    def __init__(self, n_hidden, sigma=None):
        self.n_hidden = n_hidden
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kmeans_centers(self, X):
        kmeans = KMeans(n_clusters=self.n_hidden, random_state=42).fit(X)
        centers = kmeans.cluster_centers_
        return centers

    def fit(self, X, y):
        # 选择中心
        self.centers = self._kmeans_centers(X)
        if self.sigma is None:
            # 计算带宽参数
            d_max = np.max(cdist(self.centers, self.centers))
            self.sigma = d_max / np.sqrt(2 * self.n_hidden)

        # 计算隐层输出
        G = np.zeros((X.shape[0], self.n_hidden))
        for i, x in enumerate(X):
            for j, c in enumerate(self.centers):
                G[i, j] = rbf(x, c, self.sigma)

        # 最小二乘法求解权重
        self.weights = np.linalg.pinv(G).dot(y)

    def predict(self, X):
        G = np.zeros((X.shape[0], self.n_hidden))
        for i, x in enumerate(X):
            for j, c in enumerate(self.centers):
                G[i, j] = rbf(x, c, self.sigma)
        return G.dot(self.weights)


# 使用不同的隐层节点个数训练广义RBF网络并绘制分类结构
hidden_nodes_list = [2, 3, 4]
for n_hidden in hidden_nodes_list:
    rbf_net = RBFNetwork(n_hidden=n_hidden)
    rbf_net.fit(X, y_onehot)

    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = rbf_net.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'RBF Network Decision Boundary (Hidden Nodes: {n_hidden})')
    plt.show()
