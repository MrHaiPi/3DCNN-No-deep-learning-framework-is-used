import os
import time
import numpy as np
import matplotlib.pyplot as plt
import math

import DataIo
import LayerFC
import LayerConv
import LayerMapping
import LayerMaxPool
from functools import reduce
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

class NeuralNetwork:
    def __init__(self):

        # 初始化网络
        self.layers = None

        # 初始化输出
        self.layers_pool = None

        # 保存输入数据的结构
        self.size_input_data = None

        # 保存输出层数据结构
        self.size_output_data = None

        # 网络的结构
        self.size_layer = None

    # 建立网络结构
    def build_net(self, size_input_data, size_output_data, size_layer):
        """
        :param size_input_data: 单个样本的结构,内容为[通道数，帧数，高度，宽度]
        :param size_output_data: 网络的输出结构,内容为[分类数，1]
        :param size_layer: 网络的结构(每层的名字)，ex:['conv1a','pool1','conv2a','conv2b','pool2'.....]，代表对应的层串联
        """
        self.size_input_data = size_input_data
        self.size_output_data = size_output_data
        self.size_layer = size_layer

        # 初始化网络链表
        self.layers = []
        self.layers_pool = []

        # 输入第一个全连接层的数据结构
        size_input_first_fc = self.size_input_data.copy()

        # 根据网络的结构构建网络
        for i in range(self.size_layer.__len__()):
            # 每两个池化层之间的卷积层的卷积核个数和尺寸相同
            if 'conv1' in self.size_layer[i]:
                conv_kneral = np.array([2,3,3,3])#1
                self.layers.append(LayerConv.LayerConv(conv_kneral, np.array([1, 1, 1]), np.array([1, 1, 1]), 'relu', self.size_layer[i]))
                self.layers_pool.append(LayerMapping.LayerMapping(np.array([conv_kneral[0], 1]), 'map' + self.size_layer[i][-2:]))
                # 更新卷积后的数据结构
                if 'a' in self.size_layer[i]:
                    size_input_first_fc[0] *= conv_kneral[0]
            elif 'conv2' in self.size_layer[i]:
                conv_kneral = np.array([2,3,3,3])#2
                self.layers.append(LayerConv.LayerConv(conv_kneral, np.array([1, 1, 1]), np.array([1, 1, 1]), 'relu', self.size_layer[i]))
                self.layers_pool.append(LayerMapping.LayerMapping(np.array([conv_kneral[0], 1]), 'map' + self.size_layer[i][-2:]))
                # 更新卷积后的数据结构
                if 'a' in self.size_layer[i]:
                    size_input_first_fc[0] *= conv_kneral[0]
            elif 'conv3' in self.size_layer[i]:
                conv_kneral = np.array([4,3,3,3])#4
                self.layers.append(LayerConv.LayerConv(conv_kneral, np.array([1, 1, 1]), np.array([1, 1, 1]), 'relu', self.size_layer[i]))
                self.layers_pool.append(LayerMapping.LayerMapping(np.array([conv_kneral[0], 1]), 'map' + self.size_layer[i][-2:]))
                # 更新卷积后的数据结构
                if 'a' in self.size_layer[i]:
                    size_input_first_fc[0] *= conv_kneral[0]
            elif 'conv4' in self.size_layer[i]:
                conv_kneral = np.array([4,3,3,3])#4
                self.layers.append(LayerConv.LayerConv(conv_kneral, np.array([1, 1, 1]), np.array([1, 1, 1]), 'relu', self.size_layer[i]))
                self.layers_pool.append(LayerMapping.LayerMapping(np.array([conv_kneral[0], 1]), 'map' + self.size_layer[i][-2:]))
                # 更新卷积后的数据结构
                if 'a' in self.size_layer[i]:
                    size_input_first_fc[0] *= conv_kneral[0]
            elif 'conv5' in self.size_layer[i]:
                conv_kneral = np.array([4,3,3,3])#4
                self.layers.append(LayerConv.LayerConv(conv_kneral, np.array([1, 1, 1]), np.array([1, 1, 1]), 'relu', self.size_layer[i]))
                self.layers_pool.append(LayerMapping.LayerMapping(np.array([conv_kneral[0], 1]), 'map' + self.size_layer[i][-2:]))
                # 更新卷积后的数据结构
                if 'a' in self.size_layer[i]:
                    size_input_first_fc[0] *= conv_kneral[0]
            elif 'conv6' in self.size_layer[i]:
                conv_kneral = np.array([8,3,3,3])#4
                self.layers.append(LayerConv.LayerConv(conv_kneral, np.array([1, 1, 1]), np.array([1, 1, 1]), 'relu', self.size_layer[i]))
                self.layers_pool.append(LayerMapping.LayerMapping(np.array([conv_kneral[0], 1]), 'map' + self.size_layer[i][-2:]))
                # 更新卷积后的数据结构
                if 'a' in self.size_layer[i]:
                    size_input_first_fc[0] *= conv_kneral[0]

            elif 'pool' in self.size_layer[i]:
                # 第一层池化层的池化核和步幅与其他层不同
                if '1' in self.size_layer[i] :
                    size_kernel = np.array([1,2,2])
                    step = size_kernel
                # 其他层相同
                else:
                    size_kernel = np.array([2,2,2])
                    step = size_kernel
                self.layers.append(LayerMaxPool.LayerMaxPool(size_kernel, step, self.size_layer[i]))
                self.layers_pool.append(None)
                # 更新池化后的数据结构
                size_input_first_fc[1:] = (np.array(size_input_first_fc[1:]) // np.array(step)).tolist()

            elif 'fc' in self.size_layer[i]:
                num_fc_neral = 2048
                # 与池化层或卷积层对接的全连接层初始化不同
                if 'a' in self.size_layer[i]:
                    # 池化层的所有输出全连接至全连接层
                    num_self = reduce(lambda x, y : x * y, size_input_first_fc)
                    num_next = num_fc_neral
                else:
                    num_self = num_fc_neral
                    num_next = num_fc_neral

                # 倒数第一层用softmax
                if i == self.size_layer.__len__() - 1:
                    num_next = self.size_output_data[0]
                    fun = 'softmax'
                # 其他用
                else:
                    fun = 'relu'
                self.layers.append(LayerFC.LayerFC(num_self, num_next, fun, self.size_layer[i]))
                self.layers_pool.append(None)

            # 异常
            else:
                print("网络结构输入格式错误！")
                exit()


    # 开始训练神经网络
    def train(self, train_data_path_list, train_target, test_data_path_list, test_target, batch_size = 1, epoch = 1, mu = 0.05, path = ""):
        """
        :param test_data_path_list: 测试数据文件地址
        :param test_target: 测试数据目标
        :param epoch: 所有数据总共训练的次数
        :param batch_size: 单次训练所用的训练集个数
        :param mu: 网络学习速率
        :param path: 训练完成后参数的保存位置
        :param train_data_path_list: 训练数据集地址
        :param train_target: 训练目标，结构为：[分类类别][样本编号]
        """

        # 记录上一次迭代的误差和，初始化为无穷大
        e_last = 0

        # 记录所有的误差
        e_all = []

        # 记录所有的学习速率
        mu_all = []

        # 记录每个epoch的测试集正确率
        acc_rate_test = []

        # 记录最高的正确率
        acc_rate_best = 0

        print("训练开始！")
        start = time.time()

        for i in range(epoch + 1):
            if i >= 0:
                print(">>>>>>>>>>>测试集正确率计算<<<<<<<<<<<")
                # 每个epoch进行一次正确率的计算
                rate = self.test(test_data_path_list, test_target)
                print("测试集正确率为：", (rate * 100).__round__(2), "%")
                # 保存权重
                if rate > acc_rate_best:
                    print("保存权重中...")
                    self.save_par(path)
                    acc_rate_best = rate
                #else:
                #    if np.random.random() < 0.5:
                #        print("加载最优权重中...")
                #        self.load_par(path)
                #        mu = mu / 10

                #if i % 10 == 0:
                #    mu = mu / 5

                print("学习速率为", mu)
                print("最高的正确率为：", (acc_rate_best * 100).__round__(2), "%")
                # 保存每个epoch过后的正确率
                acc_rate_test.append(rate)
                print(">>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")
            if i == epoch:
                break

            # 每一次epoch打乱数据集
            permutation = np.random.permutation(np.array(train_data_path_list).shape[0])
            shuffled_train_data_path_list = (np.array(train_data_path_list)[permutation]).tolist()
            shuffled_train_target = train_target[:, permutation]

            # 计算有多少个batch
            num_batch = train_data_path_list.__len__() // batch_size
            for j in range(num_batch):
                # 加载每个batch
                batch = DataIo.load_data_2(shuffled_train_data_path_list[j * batch_size : (j + 1) * batch_size], tuple(self.size_input_data))
                batch_target = shuffled_train_target[:, j * batch_size : (j + 1) * batch_size]

                # 以0.5的概率进行batch连续训练，目的是为了调整学习速率
                if np.random.random() < 0.5:
                    batch_epoch = 1
                else:
                    batch_epoch = 1
                # 每个batch连续训练batch_epoch次
                for k in range(batch_epoch):
                    print("============="
                          "epoch(", i + 1, "/" ,epoch , ")",
                          "batches", "(", j + 1, "/", num_batch, ")",
                          "batch_epoch", "(", k + 1, "/", batch_epoch, ")",
                            "==============")

                    # 每个batch的当前输出
                    print("前向计算中...")
                    batch_result= self.think(batch)

                    #print("神经网络输出为：")
                    #print(batch_result.round(3))
                    #print("神经网络目标为：")
                    #print(batch_target)

                    # 训练样本的交叉熵损失函数，越接近0代表与目标误差越小
                    e = -np.log(batch_result[batch_target == 1]).sum() / batch.shape[0]
                    print("与目标输出的误差为：", e)

                    #is_accept = True
                    #if k > 0:
                    #    delta_e = e - e_last
                    #    if delta_e <= 0:
                    #        is_accept = True
                    #    else:
                    #        # 模拟退火思想，以一定概率接受更差的结果
                    #        is_accept = np.random.random() < math.exp(-math.exp(delta_e / e))
#
                    #    if is_accept:
                    #        print("已接受上一次的参数调整")
                    #        mu = mu * (1 + 0)
                    #        print("学习速率增大为", mu)
                    #    else:
                    #        print("已拒绝上一次的参数调整")
                    #        mu = mu * (1 - 0)
                    #        print("学习速率减小为", mu)
                    #else:
                    #    is_accept = True
                    #    print("已接受上一次的参数调整")
                    #    print("学习速率不变为", mu)

                    is_accept = True
                    self.accept_adjust(is_accept)

                    if is_accept:
                        print("delta计算中...")
                        # 残差计算
                        delta = self.cal_delta(batch_result, batch_target)

                        # 记录本次的误差
                        e_last = e

                        e_all.append(e)
                        mu_all.append(mu)

                    else:
                        print("已跳过delta计算")

                    print("权重调整中...")
                    # 调整权重
                    self.adjust_par(batch, mu)

        end = time.time()
        print("训练完成！耗时：", end - start,"s")

        print("图像结果保存中...")
        # 绘图
        plt.plot(e_all, lw = 1, label = 'loss')
        plt.legend()
        plt.savefig('Figure/1_CV_loss.png')
        plt.close()
        np.save('Figure' + "\\" + '1_CV_loss.npy', e_all)

        plt.plot(acc_rate_test, lw = 1, label = 'acc_rate_test')
        plt.legend()
        plt.savefig('Figure/1_CV_acc_rate_test.png')
        plt.close()
        np.save('Figure' + "\\" + '1_CV_acc_rate_test.npy', acc_rate_test)

        plt.plot(mu_all, lw = 1, label = 'learn_rate')
        plt.legend()
        plt.savefig('Figure/1_CV_learn_rate.png')
        plt.close()
        np.save('Figure' + "\\" + '1_CV_learn_rate.npy', mu_all)

        # 保存训练得到的参数
        #self.save_par(path)

    # 计算测试集正确率
    def test(self, test_data_path_list, test_target):

        # 每次测试40个数据，若总数据少于等于40，则全部一次测完
        if test_data_path_list.__len__() < 20:
            step = test_data_path_list.__len__()
        else:
            step = 20

        num_right = 0

        for i in range(int(np.ceil(test_data_path_list.__len__() / step))):
            stat = i * step
            end = (i + 1 ) * step
            if end > test_data_path_list.__len__() - 1:
                end = test_data_path_list.__len__() - 1
            result = self.think(DataIo.load_data_2(test_data_path_list[stat:end], tuple(self.size_input_data)))
            temp = result[test_target[:, stat:end] == 1]
            temp1 = result.max(0) == temp
            num_right += temp1[temp1 == True].size

        return num_right / test_target.shape[1]

    # 神经网络开始思考
    def think(self, input_data):
        """
        :param input_data: 训练数据
        :return:result:网络最后一层的激活状态
        """
        result = None

        # 每层网络依次思考（信息正向传播）
        for i in range(0, self.layers.__len__()):

            # 默认输入层的上一层为池化层
            if i == 0:
                # 主干网络
                result = self.layers[i].think(input_data, 'pool')
                result = self.layers_pool[i].think(input_data, None)
            else:
                if 'conv' in self.layers[i].name:
                    # 先算池主干网络
                    result = self.layers[i].think(self.layers[i - 1].a, self.layers[i - 1].name)
                    result = self.layers_pool[i].think(self.layers[i - 1].a, None)

                elif 'pool' in self.layers[i].name:
                    result = self.layers[i].think(self.layers[i - 1].a + self.layers_pool[i - 1].a, self.layers[i - 1].name)
                else:
                    result = self.layers[i].think(self.layers[i - 1].a, self.layers[i - 1].name)

        return result

    # 计算delta
    def cal_delta(self, think_data, target_data):
        '''
        :param num_out: 网络多输出的编号
        :param think_data: 网络的前向计算输出
        :param target_data: 网络的目标输出
        :return:result:网络第一层的残差
        '''

        result = None

        # 每层网络倒序依次计算delta
        for i in range(self.layers.__len__() - 1, - 1, -1):

            if i == self.layers.__len__() - 1:
                # 输出层后一层的虚拟delta
                next_layer_delta = think_data - target_data
                # 输出层后一层的虚拟w
                next_layer_weight = np.eye(next_layer_delta.shape[0])
                next_layer_step = None
                next_layer_padding = None
                # 默认输出层后一层为全连接层
                next_layer_name = 'fc'
                next_layer_pool_position = None
                result = self.layers[i].cal_delta(next_layer_weight, next_layer_step, next_layer_padding, next_layer_delta, next_layer_name, next_layer_pool_position)

            elif 'conv' in self.layers[i + 1].name:
                next_layer_weight = self.layers[i + 1].weight
                next_layer_step = self.layers[i + 1].step
                next_layer_padding = self.layers[i + 1].padding
                next_layer_delta = self.layers[i + 1].delta
                next_layer_name = self.layers[i + 1].name
                next_layer_pool_position = None
                result = self.layers[i].cal_delta(next_layer_weight, next_layer_step, next_layer_padding, next_layer_delta, next_layer_name, next_layer_pool_position)
                result = self.layers[i].cal_delta_2(self.layers_pool[i + 1].weight, None, None, self.layers_pool[i + 1].delta, None, None)

            elif 'pool' in self.layers[i + 1].name:
                next_layer_weight = self.layers[i + 1].weight
                next_layer_step = self.layers[i + 1].step
                next_layer_padding = None
                next_layer_delta = self.layers[i + 1].delta
                next_layer_name = self.layers[i + 1].name
                next_layer_pool_position = self.layers[i + 1].pool_position

                result = self.layers[i].cal_delta(next_layer_weight, next_layer_step, next_layer_padding, next_layer_delta, next_layer_name, next_layer_pool_position)
                result = self.layers_pool[i].cal_delta(next_layer_weight, next_layer_step, next_layer_padding, next_layer_delta, next_layer_name, next_layer_pool_position)
            elif 'fc' in self.layers[i + 1].name:

                next_layer_weight = self.layers[i + 1].weight
                next_layer_step = None
                next_layer_padding = None
                next_layer_delta = self.layers[i + 1].delta
                next_layer_name = self.layers[i + 1].name
                next_layer_pool_position = None

                result = self.layers[i].cal_delta(next_layer_weight, next_layer_step, next_layer_padding, next_layer_delta, next_layer_name, next_layer_pool_position)

        return result

    # 神经网络开始调整参数
    def adjust_par(self, train_data, mu):
        """
        :param mu: 网络学习速率
        :param train_data: 训练数据
        """

        # 每层网络依次调整参数
        for i in range(self.layers.__len__()):

            # 池化层没有待调整的参数
            if 'pool' not in self.layers[i].name:

                # 默认输入层的上一层为池化层
                if i == 0:
                    self.layers[i].adjust_par(train_data, 'pool', mu)
                    self.layers_pool[i].adjust_par(train_data, 'pool', mu)
                else:
                    self.layers[i].adjust_par(self.layers[i - 1].a, self.layers[i - 1].name, mu)
                    if self.layers_pool[i] is not None and 'map' in self.layers_pool[i].name:
                        self.layers_pool[i].adjust_par(self.layers[i - 1].a, self.layers[i - 1].name, mu)

    # 是否接受上次的参数调整
    def accept_adjust(self, is_accept):
        '''
        :param is_accept: 是否接受此次参数调整
        :return:
        '''

        # 每层网络依次调整参数
        for i in range(self.layers.__len__()):
            # 池化层没有代调整的参数
            if 'pool' not in self.layers[i].name:
                self.layers[i].accept_adjust(is_accept)
                if self.layers_pool[i] is not None and 'map' in self.layers_pool[i].name:
                    self.layers_pool[i].accept_adjust(is_accept)

    # 保存参数
    def save_par(self, path):
        """
        :param path: 保存参数的子文件夹目录
        :return:
        """
        # 删除存在的文件
        root = "Parameter/" + path
        # 判断该文件夹是否存在,若存在则删除里面的所有文件，否则就创建改文件夹
        if os.path.exists(root):
            for i in os.listdir(root) :
                os.listdir(root)#返回一个列表，里面是当前目录下面的所有东西的相对路径
                file_data = root + "\\" + i#当前文件夹的下面的所有东西的绝对路径
                if os.path.isfile(file_data):
                    os.remove(file_data)
        else:
            os.mkdir(root)

        # 保存输入数据的结构
        np.save(root + "\\" + "size_input_data.npy", self.size_input_data)

        # 保存输出数据的结构
        np.save(root + "\\" + "size_output_data.npy", self.size_output_data)

        # 保存网络的结构
        np.save(root + "\\" + "size_layer.npy", self.size_layer)

        # 保存每层的参数
        for i in range(self.layers.__len__()):
            if 'pool' not in self.layers[i].name:
                self.layers[i].save_par(root)
                if self.layers_pool[i] is not None and 'map' in self.layers_pool[i].name:
                    self.layers_pool[i].save_par(root)

    # 加载参数
    def load_par(self, path = ""):
        """
        :param path: 保存参数的子文件夹目录
        :return:
        """
        root = "Parameter/" + path

        # 加载输入数据的结构
        size_input_data  = np.load(root + "\\" + "size_input_data.npy")

        # 加载输出数据的结构
        size_output_data = np.load(root + "\\" + "size_output_data.npy")

        # 加载网络的结构
        size_layer = np.load(root + "\\" + "size_layer.npy")

        # 建立网络
        self.build_net(size_input_data, size_output_data, size_layer)

        # 加载每一层的参数
        for i in range(self.layers.__len__()):
            if 'pool' not in self.layers[i].name:
                self.layers[i].load_par(root)
                if self.layers_pool[i] is not None and 'map' in self.layers_pool[i].name:
                    self.layers_pool[i].load_par(root)
