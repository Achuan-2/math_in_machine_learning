import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('click.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

# 标准化
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

# 参数初始化
theta = np.random.rand(3)

# 创建训练数据的矩阵
def to_matrix(x):
    return np.vstack([np.ones(x.size), x, x ** 2]).T

X = to_matrix(train_z)

# 预测函数
def f(x):
    return np.dot(x, theta)

# 均方误差
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)

# 学习率
ETA = 1e-3

# 误差的差值
diff = 1

# 更新次数
count = 0

# 重复学习
error = MSE(X, train_y)
while diff > 1e-2:
    # 使用随机梯度下降法更新参数
    p = np.random.permutation(X.shape[0])
    for x, y in zip(X[p,:], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x

    # 计算与上一次误差的差值
    current_error = MSE(X, train_y)
    diff = error - current_error
    error = current_error

    # 输出日志
    count += 1
    log = '第 {} 次 : theta = {}, 差值 = {:.4f}'
    print(log.format(count, theta, diff))

# 绘图确认
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()
