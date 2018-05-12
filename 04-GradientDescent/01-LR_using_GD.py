# -*- coding: UTF-8 -*-

"""
用梯度下降的优化方法来快速解决线性回归问题
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    xrange = xrange  # Python 2
except:
    xrange = range   # Python 3

# 构建数据
# 100个点
points_num = 100
# 向量list
vectors = []

# 用 Numpy 的正态随机分布函数生成 100 个点
# 这些点的（x, y）坐标值对应线性方程 y = 0.1 * x + 0.2
# 权重（Weight）为 0.1，偏差（Bias）为 0.2
# xrange 类似于 range ，只不过返回值为一个生成器，语法为：
# xrange(stop) 0 开始 步长为1
# xrange(start, stop[, step])
for i in xrange(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

# 所有的 x y 点的坐标集合
x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

# 图像 1 ：绘制出上面点的图像
plt.subplot(121)
# r* 表示点为红色星形的点
plt.plot(x_data, y_data, 'r*', label="Original data")
plt.title("Linear Regression using Gradient Descent")
# 展示标签
plt.legend()

# 使用 TensorFlow 构建线性回归模型
# 初始化 权重 Weight
# 生成均匀分布的随机值
# tf.random_uniform(
#     shape,                # 张量形状
#     minval=0,             # 最小值
#     maxval=None,          # 最大值
#     dtype=tf.float32,     # 类型
#     seed=None,            # 随机种子
#     name=None             # 操作名称
# )
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 初始化 偏差 Bias
# 创建所有元素都为0的张量
# tf.zeros(
#     shape,                # 形状
#     dtype=tf.float32,     # 类型
#     name=None             # 名称
# )
b = tf.Variable(tf.zeros([1]))
# 模型计算出来的 y
y = W * x_data + b


# 定义 loss function（损失函数）或 cost function（代价函数）
# 对 Tensor 的所有维度计算 ((y - y_data) ^ 2) 之和 / N
# y 表示模型值 y_data 表示真实值
loss = tf.reduce_mean(tf.square(y - y_data))

# 用梯度下降的优化器来最小化我们的 loss（损失）
# 设置学习率为 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)     # 让模型和实际的之间的损失 loss 尽可能少

# 创建会话
sess = tf.Session()

# 初始化数据流图中的所有变量
init = tf.global_variables_initializer()
sess.run(init)

# 训练 20 步
for step in xrange(100):
    # 优化每一步
    sess.run(train)
    # 打印出每一步的损失，权重和偏差
    print("第 {} 步的 损失 = {}, 权重 = {}, 偏差 = {}".format(
        step + 1, sess.run(loss), sess.run(W), sess.run(b)))

# 图像 2 ：绘制所有的点并且绘制出最佳拟合的直线
plt.subplot(122)
plt.plot(x_data, y_data, 'r*', label="Original data")
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data, sess.run(W) * x_data +
         sess.run(b), label="Fitted line")  # 拟合的线
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 关闭会话
sess.close()
