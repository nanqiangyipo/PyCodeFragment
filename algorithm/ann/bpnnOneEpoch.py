import numpy as np
from numpy import ndarray, random
import math


class HidenLayer:
    """ weight_matrix   m*n矩阵, m是神经元个数，n是上一层神经元个数，也就是权重的个数
        bias_vector    []，偏移量数组，包含该层所有神经元的偏移量
        active_vector   []，该层神经元的所有输出值组成的向量
        errorTerm [],该层每个神经元的误差项
    """

    def __init__(self, inputNum, nerualNum, loc=0, scale=0.1):
        if inputNum * nerualNum <= 0:
            raise ("神经网络结构不许有小于1的配置")
        # 生成一个nerualNum*inputNum的矩阵。
        self.weight_matrix = random.normal(loc=loc, scale=scale, size=inputNum * nerualNum).reshape(
            (nerualNum, inputNum))
        self.bias_vector = random.normal(loc=loc, scale=scale, size=nerualNum)
        self.active_vector = np.array([0] * nerualNum, dtype=float)
        self.errorTerm_vector = np.array([0] * nerualNum, dtype=float)

        # 累计单个样本的反向传播值
        self.weight_accumulate = np.zeros((nerualNum, inputNum))
        self.bias_accumulate = np.zeros(nerualNum)


class OutputLayer:
    """ weight_matrix   m*n矩阵, m是神经元个数，n是上一层神经元个数，也就是权重的个数
        bias_vector    []，偏移量数组，包含该层所有神经元的偏移量
        active_vector   []，该层神经元的所有输出值组成的向量
        errorTerm [],该层每个神经元的误差项
        errorTerm_vector []该层神经元的残差
    """

    def __init__(self, inputNum, nerualNum, loc=0, scale=0.1):
        if inputNum * nerualNum <= 0:
            raise ("神经网络结构不许有小于1的配置")
        # 生成一个nerualNum*inputNum的矩阵。
        self.weight_matrix = random.normal(loc=loc, scale=scale, size=inputNum * nerualNum).reshape(
            (nerualNum, inputNum))
        self.bias_vector = random.normal(loc=loc, scale=scale, size=nerualNum)
        self.active_vector = np.array([0], dtype=float)
        self.errorTerm_vector = np.array([0], dtype=float)

        # 目标值
        self.target = None

        # 累计单个样本的反向传播值
        self.weight_accumulate = np.zeros((nerualNum, inputNum))
        self.bias_accumulate = np.zeros(nerualNum)

    def setTarget(self, target):
        self.target = np.array(target)
        if self.target.dtype == '<U11':
            raise ("标记值输入错误");
            exit(0)


class InputLayer:
    def __init__(self):
        self.inputData = np.array([0])

    def setInput(self, input):
        self.inputData = np.array(input)
        if self.inputData.dtype == '<U11':
            raise ("输入值错误，检查输入层数据");
            exit(0)


class BpNN:
    # 神经网络推导参考：http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm  文中
    def __init__(self, inputDim=1, hidenConfig=[], outputDim=1, learnRate=0.1, p=0.1):
        """dim:int 输入向量维度
           hidenlayer：list 隐藏层配置列表，每一个数字代表一个隐藏层的神经元个数
           outlayer：int 输出层神经元个数
           learnRate: int 学习速率
           p：权重衰减（weight decay parameter）
        """
        # 神经网络配置信息
        self.inputDim = inputDim
        self.hidenConfig = hidenConfig
        self.outputDim = outputDim
        self.learnRate = learnRate

        # 初始化神经网络
        self.inputLayer = self.createInputlayer()
        self.hidenLayer = self.createHidenlayer(self.inputDim, self.hidenConfig)
        self.outputLayer = self.createOutputlayer(self.hidenConfig, self.outputDim)

        # 用于累积所有输入数据的期望
        self.E = 0
        # 入，平滑避免过拟合
        self.p = p

    # 创建输入层
    def createInputlayer(self):
        return InputLayer()

    # 创建隐藏层
    def createHidenlayer(self, inputDim, hidenConfig):
        if len(hidenConfig) == 0 or inputDim < 1:
            raise ("创建隐藏层错误，请检查神经网络设置");
            exit(0)
        # 复制一份隐层配置，并将输入维度插入到第一个位置
        nnConfig = hidenConfig.copy()
        nnConfig.insert(0, inputDim)
        nnConfig = np.array(nnConfig)
        if nnConfig.dtype == '<U11':
            raise ("创建隐藏层错误，神经网络配置请输入数字");
            exit(0)

        # 把融合的输入维度，隐层层配置数组转换为一层所需要的参数对，（前一层输出个数，当前层神经元个数）
        nnConfig = zip(nnConfig[0:-1], nnConfig[1:])
        hidenlayer = list()
        for config in nnConfig:
            # 创建单个隐藏
            hl = HidenLayer(config[0], config[1])
            hidenlayer.append(hl)

        return hidenlayer

    # 创建输出层
    def createOutputlayer(self, hidenConfig, outputDim):
        if len(hidenConfig) == 0 or outputDim < 1 or not str(outputDim).isnumeric():
            raise ("创建输出层错误，请检查神经网络设置");
            exit(0)
        # 因为优先检查了隐藏层的配置信息，所以不用检查隐藏层配置
        return OutputLayer(hidenConfig[-1], outputDim)

    # 前馈过程
    def feedforward(self, inputLayer, hidenLayer, outputLayer):
        # 因为第一个隐藏层的输入数据是输入层，所以需要单独优先计算
        wxMatrix = inputLayer.inputData * hidenLayer[0].weight_matrix
        # 得到该层神经元的所有wx+b值,存入wxb列表中
        wxb = wxMatrix.sum(axis=1) + hidenLayer[0].bias_vector

        # sigmoid 函数，实现输入一个list类型的数据集，返回同样顺序经过计算的np.array数据向量
        sigmoid = lambda dataList: np.array([1 / (1 + np.exp(-x)) for x in dataList])
        hidenLayer[0].active_vector = sigmoid(wxb)

        # 更新剩下的所有隐层
        if len(hidenLayer) > 1:
            for curr in range(1, len(hidenLayer)):  # curr 当前需要更新的层级索引
                pre = curr - 1
                wxMatrix = hidenLayer[pre].active_vector * hidenLayer[curr].weight_matrix
                wxb = wxMatrix.sum(axis=1) + hidenLayer[curr].bias_vector
                hidenLayer[curr].active_vector = sigmoid(wxb)

        # 更新输出层的激活值
        wxMatrix = hidenLayer[-1].active_vector * outputLayer.weight_matrix
        wxb = wxMatrix.sum(axis=1) + outputLayer.bias_vector
        outputLayer.active_vector = sigmoid(wxb)

        pass

    def updateDiff(self, hidenLayer, outputLayer):
        # 输出层f导数，fList为神经层的所有激活值，yList为目标值
        errorItem = lambda fList, yList: -fList * (1 - fList) * (yList - fList)
        # 输出层残差项计算
        outputLayer.errorTerm_vector = errorItem(outputLayer.active_vector, outputLayer.target)

        # 隐层激活函数倒数计算
        # 1、最后一层隐层的残差计算
        errorItem = lambda fList, errList, wMatrix: fList * (1 - fList) * errList.dot(wMatrix)

        hidenLayer[-1].errorTerm_vector = errorItem(hidenLayer[-1].active_vector, outputLayer.errorTerm_vector,
                                                    outputLayer.weight_matrix)

        # 剩余隐层残差计算
        if len(hidenLayer) > 1:
            for curr in reversed(range(len(hidenLayer) - 1)):  # 除了最后一层的索引，并反序
                pre = curr + 1
                hidenLayer[curr].errorTerm_vector = errorItem(hidenLayer[curr].active_vector,
                                                              hidenLayer[pre].errorTerm_vector,
                                                              hidenLayer[pre].weight_matrix)

    # 更新b偏导数
    def updateBias(self, hidenLayer, outputLayer, total):
        updateBias = lambda oldBias, accumulateBias, total: oldBias - self.learnRate * accumulateBias / total
        # 输出层
        outputLayer.bias_vector = updateBias(outputLayer.bias_vector, outputLayer.bias_accumulate, total)
        outputLayer.bias_accumulate=np.zeros(outputLayer.bias_accumulate.shape)

        # 隐藏层
        for i in range(len(hidenLayer)):
            hidenLayer[i].bias_vector = updateBias(hidenLayer[i].bias_vector, hidenLayer[i].bias_accumulate,
                                                   total)
            hidenLayer[i].bias_accumulate=np.zeros(hidenLayer[i].bias_accumulate.shape)

    # 计算更新w偏导数
    def updateWeight(self, hidenLayer, outputLayer, total):
        # 权值更新公式w=w-a*[delta(w)+p*w]  暂时没有加入惩罚因子
        updateWeight = lambda oldW, accumulateWeight, total: oldW - self.learnRate * (
        accumulateWeight / total + self.p * oldW)

        # 获得上一层的输出矩阵，行数与当前层神经元个数相同，列为上一层的输出:outputDim * hidenConfig[-1]
        outputLayer.weight_matrix = updateWeight(outputLayer.weight_matrix, outputLayer.weight_accumulate, total)
        outputLayer.weight_accumulate=np.zeros(outputLayer.weight_accumulate.shape)
        # 更新隐藏层权值
        for i in range(len(hidenLayer)):
            hidenLayer[i].weight_matrix = updateWeight(hidenLayer[i].weight_matrix, hidenLayer[i].weight_accumulate,
                                                       total)
            hidenLayer[i].weight_accumulate=np.zeros(hidenLayer[i].weight_accumulate.shape)
        pass

    # 用于批量更新，累加单个值的影响结果
    def accumulateWB(self, inputLayer, hidenLayer, outputLayer):
        # b值输出层累计
        outputLayer.bias_accumulate += outputLayer.errorTerm_vector
        # b值隐藏层累计
        for i in range(len(hidenLayer)):
            hidenLayer[i].bias_accumulate += hidenLayer[i].errorTerm_vector

        # w的输出层累计
        activeMatrix = hidenLayer[-1].active_vector.reshape([self.hidenConfig[-1], 1]).repeat(self.outputDim, axis=-1).T
        deltaW = activeMatrix * outputLayer.errorTerm_vector.reshape((self.outputDim, 1))
        outputLayer.weight_accumulate += deltaW
        # w除第一层之外的隐藏层权值
        if len(hidenLayer) > 1:
            for curr in reversed(range(1, len(hidenLayer))):
                pre = curr - 1
                activeMatrix = hidenLayer[pre].active_vector.reshape([self.hidenConfig[pre], 1]).repeat(
                    self.hidenConfig[curr], axis=-1).T
                deltaW = activeMatrix * hidenLayer[curr].errorTerm_vector.reshape((self.hidenConfig[curr], 1))
                hidenLayer[curr].weight_accumulate += deltaW
        # 隐藏层第一层的权值改变量
        activeMatrix = inputLayer.inputData.reshape([self.inputDim, 1]).repeat(self.hidenConfig[0], axis=-1).T
        deltaW = activeMatrix * hidenLayer[0].errorTerm_vector.reshape((self.hidenConfig[0], 1))
        hidenLayer[0].weight_accumulate += deltaW

        pass

    # 反向传播
    def backpropagation(self, inputLayer, hidenLayer, outputLayer):
        # 计算导数
        self.updateDiff(hidenLayer, outputLayer)

        self.accumulateWB(inputLayer, hidenLayer, outputLayer)
        # # 更新阈值
        # self.updateBias(hidenLayer, outputLayer)
        # # 更新权重
        # self.updateWeight(inputLayer, hidenLayer, outputLayer)

        pass

    def train(self, inputData, maxItera=2000):
        """ :param inputData [
                      [[......],[...]]
                      [[......],[...]]
                      ...
                      ...
                      ]
            maxItera 迭代次数

        """
        if len(inputData[0]) != 2 or len(inputData[0][0]) != self.inputDim or len(inputData[0][1]) != self.outputDim:
            raise ("输入数据与网络配置不匹配");
            exit(0)
        inputDataSize = len(inputData)
        eCalcu = lambda pridict, real, lastE: sum(0.5 * (real - pridict) ** 2) + lastE
        for i in range(maxItera):  # 训练次数
            self.E = 0
            for sample in inputData:  # 数据迭代
                self.inputLayer.setInput(sample[0])
                self.outputLayer.setTarget(sample[1])
                self.feedforward(self.inputLayer, self.hidenLayer, self.outputLayer)

                self.E = eCalcu(self.outputLayer.active_vector, self.outputLayer.target, self.E)
                self.backpropagation(self.inputLayer, self.hidenLayer, self.outputLayer)

            self.E = self.E / inputDataSize
            print(self.E)
            if self.E < 0.001:
                # print(self.E)
                break
            # 更新阈值
            self.updateBias(self.hidenLayer, self.outputLayer, inputDataSize)
            # 更新权重
            self.updateWeight(self.hidenLayer, self.outputLayer, inputDataSize)
            pass
        pass

    def test(self, testData):
        for sample in testData:
            self.inputLayer.setInput(sample[0])
            self.outputLayer.setTarget(sample[1])
            self.feedforward(self.inputLayer, self.hidenLayer, self.outputLayer)

            print(str(sample) + '==>', self.outputLayer.active_vector, end='\n')


if __name__ == '__main__':
    bpnn = BpNN(7, [200,100,200], 4, learnRate=0.5,p=0)
    data = [
        # 0  1  2  3  4  5  6
        [[1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0]],  # 0
        [[0, 0, 1, 0, 0, 1, 0], [0, 0, 0, 1]],  # 1
        [[1, 0, 1, 1, 1, 0, 1], [0, 0, 1, 0]],  # 2
        [[1, 0, 1, 1, 0, 1, 1], [0, 0, 1, 1]],  # 3
        [[0, 1, 1, 1, 0, 1, 0], [0, 1, 0, 0]],  # 4
        [[1, 1, 0, 1, 0, 1, 1], [0, 1, 0, 1]],  # 5
        [[1, 1, 0, 1, 1, 1, 1], [0, 1, 1, 0]],  # 6
        [[1, 1, 1, 0, 0, 1, 0], [0, 1, 1, 1]],  # 7
        [[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0]],  # 8
        [[1, 1, 1, 1, 0, 1, 1], [1, 0, 0, 1]]  # 9
    ]

    # bpnn = BpNN(2, [20], 1, learnRate=0.5,p=0.0000001)
    # data = [
    #     [[0, 0], [0]],
    #     [[0, 1], [1]]
    #     ,
    #     [[1, 0], [1]],
    #     [[1, 1], [0]]
    # ]
    pass
    bpnn.train(data, maxItera=30000)

    bpnn.test(data)
    pass
