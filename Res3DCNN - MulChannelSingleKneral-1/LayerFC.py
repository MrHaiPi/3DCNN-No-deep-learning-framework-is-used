import random
import numpy as np

import ActivateFunction
import Adam
import MappingFunction
import NeuralNetwork


class LayerFC:
    # 全连接层
    def __init__(self, num_of_pre_layer_neural, num_of_cur_layer_neural, act_fun_name, name):
        """
        :param num_of_pre_layer_neural: 前一层的神经元个数
        :param num_of_cur_layer_neural: 本层的神经元个数
        :param act_fun_name: 本层网络的激活函数名字
        :param name: 本层网络的名字
        """
        # 本层的激活函数
        self.act_fun = ActivateFunction.act_fun(act_fun_name)

        # 本层网络的名字
        self.name = name

        # 本层网络神经元的个数
        self.num_of_neural = int(num_of_cur_layer_neural)

        # 本层网络的输入值（状态值）
        self.z = None

        # 本层网络的输出值（激活值）
        self.a = None

        # 本层网络的delta
        self.delta = None

        # 本层的权重
        self.weight = None
        # 保存上一次更新的权重
        self.weight_last = None

        # 本层的偏置
        self.bias = None
        # 保存上一次更新的权重
        self.bias_last = None

        # 设置随机数种子
        #np.random.seed(1)
        # 初始化本神经层的权重参数
        self.weight = 2 * np.random.random((int(num_of_pre_layer_neural), int(num_of_cur_layer_neural))) - 1
        # Xavier权重初始化
        self.weight = self.weight * np.sqrt(6 / num_of_pre_layer_neural)
        # 初始化本层网络的偏置参数
        self.bias = np.zeros((self.weight.shape[1], 1))

        # Adam算法
        self.adam_w = Adam.Adam()
        self.adam_b = Adam.Adam()

    # 本层网络进行思考
    def think(self, pre_layer_a, pre_layer_name):
        """
        :param pre_layer_name: 上一层的名字
        :param pre_layer_a: 上层的激活值，为mxn的矩阵，m为上一层神经元的个数，若当前层为隐藏层第一层，则m为单次训练集的输入个数,n为训练集个数
        :return self.a:本层的激活值
        """
        # 上一层为池化层或卷积层
        if 'pool' in pre_layer_name or 'conv' in pre_layer_name:

            # 将m维数据映射为n维数据
            # pre_layer_a为5维数据，结构为：[样本编号][通道数][帧数][行][列]
            pre_layer_a_mapping = MappingFunction.mapping(pre_layer_a, (pre_layer_a[0].size, pre_layer_a.shape[0]))

            # 扩充偏执参数用于矩阵运算bias_obj的行数为本层神经元个数，列数为训练集个数
            bias_obj = np.tile(self.bias, pre_layer_a_mapping.shape[1])

            # 计算本层的状态值
            self.z = np.dot(self.weight.T, pre_layer_a_mapping) + bias_obj
            # 计算本层的激活值
            self.a = self.act_fun[0](self.z)

        # 上一层为全连接层
        elif 'fc' in pre_layer_name:

            # 扩充偏执参数用于矩阵运算bias_obj的行数为本层神经元个数，列数为训练集个数
            bias_obj = np.tile(self.bias, pre_layer_a.shape[1])

            # 计算本层的状态值
            self.z = np.dot(self.weight.T, pre_layer_a) + bias_obj
            # 计算本层的激活值
            self.a = self.act_fun[0](self.z)

        # 异常
        else:
            print(pre_layer_name + '名字格式错误')
            exit(1)

        return self.a

    # 计算delta
    def cal_delta(self, next_layer_w, next_layer_step, next_layer_padding, next_layer_delta, next_layer_name, next_layer_pool_position):
        """
        :param next_layer_padding: 下一层的padding
        :param next_layer_step: 下一层的核移动步幅
        :param next_layer_pool_position: 若下一层为池化层则需传入池化位置信息
        :param next_layer_name: 下一层的名字
        :param next_layer_w: 下一层的权重值
        :param next_layer_delta:下一层的delta
        :return:本层网络的残差
        """

        # 下一层为池化层
        if 'pool' in next_layer_name:
            print('未提供此结构的残差计算公式：fc-pool')
            exit(1)

        # 下一层为卷积层
        elif 'conv' in next_layer_name:
            print('未提供此结构的残差计算公式：fc-conv')
            exit(1)

        # 下一层为全连接层
        elif 'fc' in next_layer_name:
            # 计算所有训练集的delta
            self.delta = np.dot(next_layer_w, next_layer_delta) * self.act_fun[1](self.a)

        # 异常
        else:
            print(next_layer_name + '名字格式错误')
            exit(1)

        return self.delta

    # 本层网络参数进行调整
    def adjust_par(self, pre_layer_a, pre_layer_name, mu):
        """
        :param mu: 学习速率
        :param pre_layer_name: 上一层的名字
        :param pre_layer_a: 上一层的激活值
        :return:
        """

        # 保存更新前的权重
        #self.weight_last = self.weight.copy()
        #self.bias_last = self.bias.copy()


        # 上一层为池化层或卷积层
        if 'pool' in pre_layer_name or 'conv' in pre_layer_name:

            # 将m维数据映射为n维数据
            # pre_layer_a为5维数据，结构为：[样本编号][通道数][帧数][行][列]
            pre_layer_a_mapping = MappingFunction.mapping(pre_layer_a, (pre_layer_a[0].size, pre_layer_a.shape[0]))

            # 调整w参数,若要考虑平均代价，可除以self.delta.shape[1]（训练集个数）
            w_delta = 1 / self.z.shape[1] * np.dot(pre_layer_a_mapping, self.delta.T)
            # 调整b参数
            b_delta =  1 / self.z.shape[1] * np.dot(self.delta, np.ones((self.delta.shape[1], 1)))

            w_delta = self.adam_w.cal_delta(w_delta)
            b_delta = self.adam_b.cal_delta(b_delta)

            self.weight -= mu * w_delta
            self.bias -= mu * b_delta

        # 上一层为全连接层
        elif 'fc' in pre_layer_name:
            # 调整w参数,若要考虑平均代价，可除以self.delta.shape[1]（训练集个数）
            w_delta = 1 / self.z.shape[1] * np.dot(pre_layer_a, self.delta.T)
            # 调整b参数
            b_delta =  1 / self.z.shape[1] * np.dot(self.delta, np.ones((self.delta.shape[1], 1)))

            w_delta = self.adam_w.cal_delta(w_delta)
            b_delta = self.adam_b.cal_delta(b_delta)

            self.weight -= mu * w_delta
            self.bias -= mu * b_delta
        # 异常
        else:
            print(pre_layer_name + '名字格式错误')
            exit(1)

    # 是否接受此次的参数调整
    def accept_adjust(self, is_accept):
        '''
        :param is_accept: 是否接受此次参数调整
        :return:
        '''

        if not is_accept:
            self.weight = self.weight_last.copy()
            self.bias = self.bias_last.copy()


    # 保存本层网络的参数
    def save_par(self, path):
        """
        :param path: 保存参数的子文件夹目录
        :return:
        """
        np.save(path + "\\" + self.name + "_weight.npy",self.weight)
        np.save(path + "\\" + self.name + "_bias.npy",self.bias)

    # 加载本层网络的参数
    def load_par(self, path):
        """
        :param path: 保存参数的子文件夹目录
        :return:
        """
        self.weight = np.load(path + "\\" + self.name + "_weight.npy")
        self.bias = np.load(path + "\\" + self.name + "_bias.npy")
