"只一个两类的分类器"

import tensorflow as tf
from PIL import Image
import os
import numpy as np

from numpy.random.mtrand import shuffle

import datetime as time
# 图片大小，神经网络第一层能否实现任意图像大小的输入？
# 现在resize 了图像，必然对实际应用有影响

# 狗 ==> [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# 鬣狗 ==> [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# 花栗鼠 ==> [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# 狐狸 ==> [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
# 天竺鼠 ==> [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
# 黄鼠狼 ==> [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
# 松鼠 ==> [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
# 驯鹿 ==> [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
# 猫 ==> [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]
# 长颈鹿 ==> [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
# 梅花鹿 ==> [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
# 狼 ==> [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]

# 读出所有图片，随机打乱，上标签
# image size
# width,height,channel= 600,400,3
width,height,channel= 150,100,3



class Traindata:
    def getImagePaths(self, par_path):
        """获取图片文件全路径"""
        return [os.path.join(par_path, filename) for filename in os.listdir(par_path)]

    # 数据格式：
    # [
    #   [
    #     [image:[height400,width600,channel3]],
    #     [label:12]
    #   ],
    # ,
    # ,
    # ]
    def __init__(self, data_path=r'/home/xinye/workingdirectory/PyCodeFragment/data/resized_animal', splitline=0.9):
        """这个版本直将图片载入内存，对于4g图片，考虑一次只缓存图片的全路径，feed之前再读入图片"""
        # 各个图片文件夹的名字
        animal_path_name = os.listdir(data_path)

        # 生成每个类别的编码
        labels = np.zeros((len(animal_path_name), len(animal_path_name)))
        for i in range(len(labels)):
            labels[i][i] = 1
            print(animal_path_name[i], '==>', labels[i])

        # 图片文件夹全路径
        animal_paths = [os.path.join(data_path, filename) for filename in animal_path_name]

        self.train_data = list()
        for i in range(len(animal_paths)):
            image_paths = self.getImagePaths(animal_paths[i])
            print('获得%s' % animal_path_name[i])
            for filename in image_paths:
                pic = Image.open(filename)
                self.train_data.append([np.asarray(pic).flatten(), labels[i]])
                if len(self.train_data[-1][0]) != height * width * channel:
                    print('图片大小异常（检查图片格式）%d--%s' % (len(self.train_data[-1][0]), filename))
                pic.close()
        print('......\n图片提取完成\n')

        # 转换成numpy并打乱顺序
        self.train_data = np.array(self.train_data)
        print(self.train_data.shape)
        self.splitline = splitline
        self.spliDtata()

        # 取数据标记
        self.flag = 0

    # 训练数据小批量获取
    def next_batch(self, batch_size):
        if len(self.train_data) > 0 and self.flag < len(self.train_data):
            self.flag += batch_size
            traindata = self.train_data[self.flag - batch_size:self.flag, 0].tolist()
            trainlabel = self.train_data[self.flag - batch_size:self.flag, 1].tolist()

            return traindata, trainlabel
        else:
            return None, None

    # 获得全部测试数据
    def get_testdata(self):
        return self.test_data[:, 0].tolist(), self.test_data[:, 1].tolist()

    def getAll(self):
        traindata = self.train_data[:, 0].tolist()
        trainlabel = self.train_data[:, 1].tolist()
        return traindata , trainlabel
    def getSlice(self,start,length):
        traindata = self.train_data[start:start+length, 0].tolist()
        trainlabel = self.train_data[start:start+length, 1].tolist()
        return traindata , trainlabel

    def isOutsize(self):
        if self.flag>=len(self.train_data):
            return True
        return False
    def restFlag(self):
        # self.spliDtata()
        self.flag=0

    def spliDtata(self):
        shuffle(self.train_data)

        # 切分出测试样本和训练样本
        split_index = int(np.ceil(len(self.train_data) * self.splitline))
        self.test_data = self.train_data[split_index:]
        self.train_data = self.train_data[:split_index]

# 神经网络训练
# Parameters
learning_rate = 0.005
training_iters = 100000
batch_size = 170
display_step = 4

# Network Parameters
n_classes = 2  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units
fc1=128
# design the neual network
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),

    'wc3': tf.Variable(tf.random_normal([2, 2, 64, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([int(np.ceil(height/4)*np.ceil(width/4)*64), fc1])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([fc1, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([fc1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, [None, height * width * channel])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


# 对内置函数封装一下
# 卷积函数封装
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# 向下采样函数封装
def maxpool2d(x, k=2,s=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding='SAME')

def collectImage(name,tensor,num=5):
    temp = tf.reduce_mean(tensor,axis=3,keep_dims=True)
    tf.image_summary(tag=name,tensor=temp,max_images=num)

# 网络构建
def conv_net(x1, weights, biases, dropout):
    x = tf.reshape(x1, shape=[-1, height, width, channel])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    collectImage('conv1',conv1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    collectImage('conv1pool', conv1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    collectImage('conv2', conv2)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    collectImage('conv2pool', conv2)


    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    collectImage('conv3', conv3)

    # Max Pooling (down-sampling)
    # conv3 = maxpool2d(conv3, k=2)
    #
    # conv3maxout = tf.reduce_mean(conv3,axis=3,keep_dims=True)
    # tf.image_summary(tag='conv3max',tensor=conv3maxout,max_images=5)

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
tf.histogram_summary('cost',cost,name='cost')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    data=Traindata(data_path=r'/home/xinye/workingdirectory/PyCodeFragment/data/test_data',splitline=0.95)
    # data = Traindata(data_path=r'/home/xinye/workingdirectory/PyCodeFragment/data/resized_animal',splitline=0.95)
    testdata = data.get_testdata()
    merged_summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(r'/home/xinye/workingdirectory/PyCodeFragment/data/tensorboard', sess.graph)

    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    # while step * batch_size < training_iters:
    flag = 1
    while True:
        batch_x, batch_y = data.next_batch(batch_size)
        if data.isOutsize():
            # print('搞完一次了')
            data.restFlag()
        # a,b=data.getSlice(4864,batch_size)
        # a=np.asarray(a)
        # b=np.asarray(b)
        # Run optimization op (backprop)
        startTime=time.datetime.now()

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        endTime=time.datetime.now()


        if step % display_step == 0:
            # loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
            #                                                   y: batch_y,
            #                                                   keep_prob: 1.})
            # print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
            #       "{:.6f}".format(loss) + ", Training Accuracy= " + \
            #       "{:.5f}".format(acc), step)
            # Calculate batch loss and accuracy
            # print("%dTestingAccuracy:"%step, sess.run([accuracy,cost], feed_dict={x: testdata[0],y: testdata[1], keep_prob: 1.}))
            acc,coss = sess.run([accuracy,cost], feed_dict={x: testdata[0],y: testdata[1], keep_prob: 1.})
            summstr=sess.run(merged_summary_op,feed_dict={x: testdata[0],y: testdata[1], keep_prob: 1.})
            str = summary_writer.add_summary(summstr, step)
            print(coss,acc)
            if coss<34901 and flag==0:
                # print(sess.run(pred,feed_dict={x: testdata[0],y: testdata[1], keep_prob: 1.}))
                predfile = open(r'/home/xinye/workingdirectory/PyCodeFragment/data/predfile.txt',mode='w',encoding='utf8')
                predfile.write(str(sess.run(pred,feed_dict={x: testdata[0],y: testdata[1], keep_prob: 1.}).tolist()))
                predfile.flush()
                predfile.close()

                # print(list(testdata[1]))
                labelfile = open(r'/home/xinye/workingdirectory/PyCodeFragment/data/labelfile.txt',mode='w',encoding='utf8')
                labelfile.write(str(list(testdata[1])))
                labelfile.flush()
                labelfile.close()
                flag=1
        step += 1

    # Calculate accuracy for 256 mnist test images

    # a=np.asarray(testdata[0],dtype=np.float)
    # b=np.asarray(testdata[1],dtype=np.float)
    # print(a.shape,b.shape)
