# to get the data path in these project
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

# mnist数据集的目录，先获得整个工程中数据目录的根目录，然后拼接子目录
dataPathRoot = os.path.abspath('./../../../data')
mnistDataPath = os.path.join(dataPathRoot, 'MNIST_data')
print(mnistDataPath)
# get the mnist train data
mnist = input_data.read_data_sets(mnistDataPath, one_hot=True)

batch_size=100
# 训练或测试数据输入
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 28 * 28], name='input_x')
    y = tf.placeholder(tf.float32, [None, 10], name='target_y')

# 需要优化的变量
w = tf.Variable(tf.zeros([28 * 28, 10]), name='weight')
b = tf.Variable(tf.zeros(10), name='bias')

# 线性变化 + softmax
pred = tf.nn.softmax(tf.matmul(x, w) + b, name='softmax')


# 交叉熵和梯度下降优化
cross_entropy=-tf.reduce_sum(y*tf.log(pred))
train_step=tf.train.GradientDescentOptimizer(0.01, name='GradientDescent').minimize(cross_entropy)

# 模型评估
correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuray=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init_variable = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_variable)


    # 训练
    # for i in range(1000):
    #     one_batch_x, one_batch_y = mnist.train.next_batch(batch_size)
    #     sess.run(train_step, feed_dict={x: one_batch_x.astype(np.float32), y: one_batch_y.astype(np.float32)})
    one_batch_x, one_batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: one_batch_x, y: one_batch_y})

    # 测试
    print(sess.run(accuray,feed_dict={x:mnist.test.images[:batch_size],y:mnist.test.labels[:batch_size]}))
