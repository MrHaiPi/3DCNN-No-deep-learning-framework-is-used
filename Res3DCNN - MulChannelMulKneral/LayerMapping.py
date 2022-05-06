import numpy as np

import ActivateFunction
import Adam
import ConvolutionFunction
import MappingFunction
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import PoolFunction


class LayerMapping:
    def __init__(self, name):
        """
        :param name: 本层网络的名字
        """

        # 本层网络的名字
        self.name = name

        # 本层网络的输入值（状态值）
        self.z = None

        # 本层网络的输出值（激活值）
        self.a = None

        # 本层网络的delta
        self.delta = None

        # 本层的权重
        self.weight = None
        self.weight_last = None

        # 设置随机数种子
        #np.random.seed(1)
        # 初始化本神经层的权重参数
        self.weight = 2 * np.random.random() - 1
        # Adam算法
        self.adam_w = Adam.Adam()

    # 本层网络进行思考
    def think(self, pre_layer_a, pre_layer_name):
        """
        :param pre_layer_name: 上一层的名字
        :param pre_layer_a: 上层的激活值，为mxn的矩阵，m为上一层神经元的个数，若当前层为隐藏层第一层，则m为单次训练集的输入个数,n为训练集个数
        :return self.a:本层的激活值
        """
        self.z = pre_layer_a * self.weight
        self.a = self.z.copy()

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
            # 初始化delta张量的大小
            self.delta = np.zeros(self.z.shape)

            # 多核并行计算
            #task = [[0 for t in range(self.delta.shape[1])]for i in range(self.delta.shape[0])]
            #pool = ProcessPoolExecutor(max_workers = cpu_count())
            ## 对每个样本进行计算
            #for i in range(self.delta.shape[0]):
            #    # 对每个通道进行计算
            #    for j in range(self.delta.shape[1]):
            #        task[i][j] = pool.submit(PoolFunction.three_dem_upsample, next_layer_w, next_layer_step, next_layer_delta[i, j], next_layer_pool_position[i, j].astype(int))
            #for i in range(self.delta.shape[0]):
            #    # 对每个通道进行计算
            #    for j in range(self.delta.shape[1]):
            #        self.delta[i, j] = task[i][j].result()
            #pool.shutdown(wait = True)


            # 串行计算
            # 对每个样本进行计算
            for i in range(self.delta.shape[0]):
                # 对每个通道进行计算
                for j in range(self.delta.shape[1]):
                    self.delta[i, j] = PoolFunction.three_dem_upsample(next_layer_w, next_layer_step, next_layer_delta[i, j], next_layer_pool_position[i, j].astype(int))

        # 下一层为卷积层
        elif 'conv' in next_layer_name:
            # 初始化delta张量的大小
            self.delta = np.zeros(self.z.shape)
            # 将2，3维度的数据旋转180度
            nex_layer_w_r180 = np.rot90(next_layer_w, 2, (2, 3))

            # 多核并行计算
            #task = [[[0 for k in range(nex_layer_w_r180.shape[0])] for t in range(self.delta.shape[1])]for i in range(self.delta.shape[0])]
            #pool = ProcessPoolExecutor(max_workers = cpu_count())
            ## 对每个样本进行计算
            #for i in range(self.delta.shape[0]):
            #    # 对每个通道进行计算
            #    for j in range(self.delta.shape[1]):
            #        for k in range(nex_layer_w_r180.shape[0]):
            #            # 放入进程池
            #            task[i][j][k] = pool.submit(ConvolutionFunction.three_dem_convolution, nex_layer_w_r180[k], next_layer_step, next_layer_padding, next_layer_delta[i, j * nex_layer_w_r180.shape[0] + k])
            ## 对每个样本进行计算
            #for i in range(self.delta.shape[0]):
            #    # 对每个通道进行计算
            #    for j in range(self.delta.shape[1]):
            #        for k in range(nex_layer_w_r180.shape[0]):
            #            self.delta[i, j] += task[i][j][k].result()
            #pool.shutdown(wait = True)


            # 串行计算
            # 对每个样本进行计算
            for i in range(self.delta.shape[0]):
                for j in range(self.delta.shape[1]):
                    for k in range(nex_layer_w_r180.shape[0]):
                       self.delta[i, j] += ConvolutionFunction.three_dem_convolution(nex_layer_w_r180[k], next_layer_step, next_layer_padding, next_layer_delta[i, j * nex_layer_w_r180.shape[0] + k])

        # 下一层为全连接层
        elif 'fc' in next_layer_name:

            # 初始化delta张量的大小
            self.delta = np.zeros(self.z.shape)

            # temp 为2维数据，第一个维度为样本的信息，第二个维度为样本编号
            temp = np.dot(next_layer_w, next_layer_delta)

            # 低维映射到高维
            self.delta = MappingFunction.unmapping(temp, self.delta.shape)

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
        #self.weight_last = self.weight

        pre_layer_a_temp = self.z / self.weight
        w_delta = 1 / self.z.shape[0] * (self.delta * pre_layer_a_temp).sum()
        w_delta = self.adam_w.cal_delta(w_delta)
        self.weight -= mu * w_delta

    # 是否接受此次的参数调整
    def accept_adjust(self, is_accept):
        '''
        :param is_accept: 是否接受此次参数调整
        :return:
        '''

        if not is_accept:
            self.weight = self.weight_last


    # 保存本层网络的参数
    def save_par(self, path):
        """
        :param path: 保存参数的子文件夹目录
        :return:
        """
        np.save(path + "\\" + self.name + "_weight.npy",self.weight)

    # 加载本层网络的参数
    def load_par(self, path):
        """
        :param path: 保存参数的子文件夹目录
        :return:
        """
        self.weight = np.load(path + "\\" + self.name + "_weight.npy")
