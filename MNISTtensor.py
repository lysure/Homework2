import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 网络结构设计
# 1. 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 数据预处理
# 将输入数据reshape为 (60000, 784)，并归一化到 [0, 1] 范围
x_train = x_train.reshape(60000, 28 * 28).astype('float32') / 255
x_test = x_test.reshape(10000, 28 * 28).astype('float32') / 255
# 将标签进行one-hot编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. 构建BP神经网络模型
# 网络结构设计: 使用两层隐藏层的BP神经网络，输入层有784个节点，第一隐藏层有128个节点，第二隐藏层有64个节点，输出层有10个节点，对应数字0-9。
model = Sequential()
model.add(Dense(128, input_shape=(784,), activation='relu'))  # 输入层和第一隐藏层
model.add(Dense(64, activation='relu'))  # 第二隐藏层
model.add(Dense(10, activation='softmax'))  # 输出层

# 训练方法
# 4. 编译模型
# 使用交叉熵损失函数和Adam优化器进行训练，学习率设置为默认值。
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. 训练模型
# 使用MNIST数据集训练模型，训练10个epoch，每个batch大小为128。
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# 识别结果
# 6. 评估模型
# 在测试集上评估模型的性能，计算并输出测试集的准确率。
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'测试集上的准确率: {test_acc * 100:.2f}%')