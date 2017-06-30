import tensorflow as tf
from PIL import Image
import os
import numpy as np

from numpy.random.mtrand import shuffle
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

class Traindata:

    def getImagePaths(self,par_path):
        """获取图片文件全路径"""
        return [os.path.join(par_path,filename) for filename in os.listdir(par_path)]

    # 数据格式：
    # [
    #   [
    #     [image:[height400,width600,channel3]],
    #     [label:12]
    #   ],
    # ,
    # ,
    # ]
    def __init__(self,data_path=r'/home/xinye/PycharmProjects/PyCodeFragment/data/resized_animal',splitline=0.8):
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

        self.train_data=list()
        for i in range(len(animal_paths)):
            image_paths=self.getImagePaths(animal_paths[i])
            print('获得%s'%animal_path_name[i])
            for filename in image_paths:
                pic=Image.open(filename)
                self.train_data.append([np.asarray(pic).flatten(),labels[i]])
                pic.close()
        print('......\n图片提取完成\n')

        # 转换成numpy并打乱顺序
        self.train_data=np.array(self.train_data)
        print(self.train_data.shape)
        shuffle(self.train_data)

        # 切分出测试样本和训练样本
        split_index=int(np.ceil(len(self.train_data)*splitline))
        self.test_data=self.train_data[split_index:]
        self.train_data=self.train_data[:split_index]

        # 取数据标记
        self.flag=0
    # 训练数据小批量获取
    def next_batch(self,batch_size):
        if len(self.train_data)>0 and self.flag<len(self.train_data):
            self.flag+=batch_size
            traindata=self.train_data[self.flag-batch_size:self.flag,0].tolist()
            trainlabel=self.train_data[self.flag-batch_size:self.flag,1].tolist()
            pass
            return traindata,trainlabel
        else:
            return None,None

    # 获得全部测试数据
    def get_testdata(self):
        return self.test_data[:,0],self.test_data[:,1]



# image size
width = 600
height = 400
channel =3


x=tf.placeholder(tf.float32,[None,height*width*channel])

y=tf.reshape(x,shape=[-1, height, width, channel])

with tf.Session() as sess:
    data=Traindata(data_path=r'/home/xinye/PycharmProjects/PyCodeFragment/data/test_data')
    # data = Traindata()
    batchx,batchy=data.next_batch(100)
    b=np.asarray(batchx)
    print(b.shape)
    sess.run(y,feed_dict={x:batchx})