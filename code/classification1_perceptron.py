import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('images1.csv', delimiter=',', skiprows=1)
train_x = train[:,0:2]
train_y = train[:,2]

# 权重初始化
w = np.random.rand(2)

# 判别函数
def f(x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1

# 重复次数
epoch = 10

# 更新次数
count = 0

# 学习权重
for _ in range(epoch):
    for x, y in zip(train_x, train_y):
        if f(x) != y:
            w = w + y * x

            # 输出日志
            count += 1
            print('第 {} 次 : w = {}'.format(count, w))

# 绘图确认
x1 = np.arange(0, 500)
plt.plot(train_x[train_y ==  1, 0], train_x[train_y ==  1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.plot(x1, -w[0] / w[1] * x1, linestyle='dashed')
plt.show()
