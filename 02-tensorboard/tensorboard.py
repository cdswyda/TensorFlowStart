# -*- coding: UTF-8 -*-

# 引入tensorflow
import tensorflow as tf

# 构造图（Graph）的结构
# 用一个线性方程的例子 y = W * x + b
W = tf.Variable(2.0, dtype=tf.float32, name="Weight") # 权重
b = tf.Variable(1.0, dtype=tf.float32, name="Bias") # 偏差
x = tf.placeholder(dtype=tf.float32, name="Input") # 输入
with tf.name_scope("Output"):      # 输出的命名空间
    y = W * x + b    # 输出

#const = tf.constant(2.0) # 不需要初始化

# 定义保存日志的路径
path = "./log"

# 创建用于初始化所有变量（Variable）的操作
init = tf.global_variables_initializer()

# 创建Session（会话）
with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 创建写文件的writer
    writer = tf.summary.FileWriter(path, sess.graph)
    result = sess.run(y, {x: 3.0})
    # 打印 y = W * x + b 的值，就是 7
    print("y = %s" % result) 
    result = sess.run(y, {x: 4.0})
    # 打印 y = W * x + b 的值，就是 9
    print("y = %s" % result) 
    result = sess.run(y, {x: 5.0})
    print("y = %s" % result) 
    result = sess.run(y, {x: 6.0})
    print("y = %s" % result) 
