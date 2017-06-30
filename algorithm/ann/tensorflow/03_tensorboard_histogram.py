from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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

batch_size=200
# 训练或测试数据输入
with tf.name_scope('1_0input'):
    x = tf.placeholder(tf.float32, [None, 28 * 28], name='input_x')
    y = tf.placeholder(tf.float32, [None, 10], name='target_y')

# 需要优化的变量
with tf.name_scope('1_1linefuanction'):
    w = tf.Variable(tf.zeros([28 * 28, 10]), name='weight')
    tf.histogram_summary('w',w,name='weight')
    b = tf.Variable(tf.zeros(10), name='bias')
    tf.histogram_summary('b',b,name='bias')
    xwb=tf.matmul(x, w) + b

with tf.name_scope('2_0softmax'):
    tf.histogram_summary('xwb',xwb)
    pred = tf.nn.softmax(xwb, name='softmax')
    tf.histogram_summary('pred',pred)

# 交叉熵和梯度下降优化
with tf.name_scope('3_0cross_entropy'):
    cross_entropy=-tf.reduce_sum(y*tf.log(pred))
    # tf.histogram_summary('cross_entropy',cross_entropy)
    train_step=tf.train.GradientDescentOptimizer(0.01, name='GradientDescent').minimize(cross_entropy)

# 模型评估
correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuray=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.scalar_summary('accuray',accuray)

# 创建graph管理器
sess = tf.Session()

# 初始化图中的变量
init_variable = tf.initialize_all_variables()
sess.run(init_variable)

#
merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(r'/home/xinye/workingdirectory/tensorflow_logs',sess.graph)


for step in range(100):
    one_batch_x, one_batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: one_batch_x, y: one_batch_y})
    if step%20==0:
        summary_str = sess.run(merged_summary_op,feed_dict={x: one_batch_x, y: one_batch_y})
        str=summary_writer.add_summary(summary_str,step)
        print(sess.run(accuray,feed_dict={x:mnist.test.images,y:mnist.test.labels}))



summary_writer.close()
sess.close()