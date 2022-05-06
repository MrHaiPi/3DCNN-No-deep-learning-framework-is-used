import random
import numpy as np

import ActivateFunction
import Adam
import ConvolutionFunction
import MappingFunction
import NeuralNetwork
import PoolFunction
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

class LayerConv:
    # 卷积层
    def __init__(self, size_of_kernel, step, padding, act_fun_name, name):
        """
        :param size_of_kernel: 核尺寸，内容为：[个数，时间域长度，空间域高度，空间域宽度]
        :param step: 步长信息，内容为：[时间域步长，空间域高度步长，空间域宽度步长]
        :param padding: 填充信息，内容为：[时间域填充0个数/2，空间域高度填充0个数/2，空间域宽度填充0个数/2]
        :param act_fun_name: 本层网络的激活函数
        :param name: 本层网络的名字
        """
        # 卷积padding
        self.padding = padding

        # 卷积步长
        self.step = step

        # 本层的激活函数
        self.act_fun = ActivateFunction.act_fun(act_fun_name)

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
        # 保存上一次更新的权重
        self.weight_last = None

        # 本层的偏置
        self.bias = None
        # 保存上一次更新的权重
        self.bias_last = None

        # 设置随机数种子
        np.random.seed(2)
        # 初始化本神经层的权重参数
        self.weight = 2 * np.random.random(size_of_kernel) - 1
        # He权重初始化
        self.weight = self.weight * (np.sqrt(6 / self.weight[0].size))
        # 初始化本层网络的偏置参数
        self.bias = np.zeros(self.weight.shape[0])
        # Adam算法
        self.adam_w = Adam.Adam()
        self.adam_b = Adam.Adam()

    # 本层网络进行思考
    def think(self, input_data, pre_layer_name):
        """
        :param pre_layer_name: 上一层的名字
        :param input_data: 输入数据，为5维数组，结构为：input_data[样本编号][通道数][帧数][行][列]
        :return:本层网络的激活态
        """

        # 上一层为池化层
        if 'pool' in pre_layer_name:
            # 初始化状态张量和激活张量的大小
            self.z = np.zeros((input_data.shape[0],
                               input_data.shape[1] * self.weight.shape[0],
                               (input_data.shape[2] + 2 * self.padding[0] - self.weight.shape[1]) // self.step[0] + 1,
                               (input_data.shape[3] + 2 * self.padding[1] - self.weight.shape[2]) // self.step[1] + 1,
                               (input_data.shape[4] + 2 * self.padding[2] - self.weight.shape[3]) // self.step[2] + 1))
            self.a = np.zeros(self.z.shape)
            # 计算最新的状态和激活值

            # 多核并行计算
            #task = [[0 for t in range(self.z.shape[1])]for i in range(self.z.shape[0])]
            #pool = ProcessPoolExecutor(max_workers = cpu_count())
            #for i in range(input_data.shape[0]):
            #    # 对每个通道进行计算
            #    for j in range(input_data.shape[1]):
            #        for k in range(self.weight.shape[0]):
            #            task[i][j * self.weight.shape[0] + k] = pool.submit(ConvolutionFunction.three_dem_convolution, self.weight[k], self.step, self.padding, input_data[i, j])
            ## 对每个样本进行计算
            #for i in range(input_data.shape[0]):
            #    # 对每个通道进行计算
            #    for j in range(input_data.shape[1]):
            #        for k in range(self.weight.shape[0]):
            #            # 计算本层的状态值
            #            self.z[i, j * self.weight.shape[0] + k] = task[i][j * self.weight.shape[0] + k].result() + self.bias[k]
            #            # 计算本层的激活值
            #            self.a[i, j * self.weight.shape[0] + k] = self.act_fun[0](self.z[i, j * self.weight.shape[0] + k])
            #pool.shutdown(wait = True)
            #'''


            # 串行计算
            # 对每个样本进行计算
            for i in range(input_data.shape[0]):
                # 对每个通道进行计算
                for j in range(input_data.shape[1]):
                    for k in range(self.weight.shape[0]):
                        # 计算本层的状态值
                        self.z[i, j * self.weight.shape[0] + k] = ConvolutionFunction.three_dem_convolution(self.weight[k], self.step, self.padding, input_data[i, j]) + self.bias[k]
                        # 计算本层的激活值
                        self.a[i, j * self.weight.shape[0] + k] = self.act_fun[0](self.z[i, j * self.weight.shape[0] + k])


        # 上一层为卷积层
        elif 'conv' in pre_layer_name:

            # 一对一连接
            if self.name[-2] == pre_layer_name[-2]:
                # 初始化状态张量和激活张量的大小
                self.z = np.zeros((input_data.shape[0],
                                   input_data.shape[1],
                                   (input_data.shape[2] + 2 * self.padding[0] - self.weight.shape[1]) // self.step[0] + 1,
                                   (input_data.shape[3] + 2 * self.padding[1] - self.weight.shape[2]) // self.step[1] + 1,
                                   (input_data.shape[4] + 2 * self.padding[2] - self.weight.shape[3]) // self.step[2] + 1))
                self.a = np.zeros(self.z.shape)
                # 计算最新的状态和激活值

                # 多核并行计算
                #task = [[0 for t in range(self.z.shape[1])]for i in range(self.z.shape[0])]
                #pool = ProcessPoolExecutor(max_workers = cpu_count())
                #for i in range(input_data.shape[0]):
                #    # 对每个通道进行计算
                #    for j in range(input_data.shape[1]):
                #        num = j % self.weight.shape[0]
                #        task[i][j] = pool.submit(ConvolutionFunction.three_dem_convolution, self.weight[num], self.step, self.padding, input_data[i, j])
                #for i in range(input_data.shape[0]):
                #    # 对每个通道进行计算
                #    for j in range(input_data.shape[1]):
                #        num = j % self.weight.shape[0]
                #        # 计算本层的状态值
                #        self.z[i, j] = task[i][j].result() + self.bias[num]
                #        # 计算本层的激活值
                #        self.a[i, j] = self.act_fun[0](self.z[i, j])
                #pool.shutdown(wait = True)
                #'''


                # 串行计算
                # 对每个样本进行计算
                for i in range(input_data.shape[0]):
                    # 对每个通道进行计算
                    for j in range(input_data.shape[1]):
                        num = j % self.weight.shape[0]
                        # 计算本层的状态值
                        self.z[i, j] = ConvolutionFunction.three_dem_convolution(self.weight[num], self.step, self.padding, input_data[i, j]) + self.bias[num]
                        # 计算本层的激活值
                        self.a[i, j] = self.act_fun[0](self.z[i, j])


            # 多对一连接
            else:
                # 初始化状态张量和激活张量的大小
                self.z = np.zeros((input_data.shape[0],
                                   input_data.shape[1] * self.weight.shape[0],
                                   (input_data.shape[2] + 2 * self.padding[0] - self.weight.shape[1]) // self.step[0] + 1,
                                   (input_data.shape[3] + 2 * self.padding[1] - self.weight.shape[2]) // self.step[1] + 1,
                                   (input_data.shape[4] + 2 * self.padding[2] - self.weight.shape[3]) // self.step[2] + 1))
                self.a = np.zeros(self.z.shape)
                # 计算最新的状态和激活值

                # 多核并行计算
                #task = [[0 for t in range(self.z.shape[1])]for i in range(self.z.shape[0])]
                #pool = ProcessPoolExecutor(max_workers = cpu_count())
                #for i in range(input_data.shape[0]):
                #    # 对每个通道进行计算
                #    for j in range(input_data.shape[1]):
                #        for k in range(self.weight.shape[0]):
                #            task[i][j * self.weight.shape[0] + k] = pool.submit(ConvolutionFunction.three_dem_convolution, self.weight[k], self.step, self.padding, input_data[i, j])
                ## 对每个样本进行计算
                #for i in range(input_data.shape[0]):
                #    # 对每个通道进行计算
                #    for j in range(input_data.shape[1]):
                #        for k in range(self.weight.shape[0]):
                #            # 计算本层的状态值
                #            self.z[i, j * self.weight.shape[0] + k] = task[i][j * self.weight.shape[0] + k].result() + self.bias[k]
                #            # 计算本层的激活值
                #            self.a[i, j * self.weight.shape[0] + k] = self.act_fun[0](self.z[i, j * self.weight.shape[0] + k])
                #pool.shutdown(wait = True)
                #'''


                # 串行计算
                # 对每个样本进行计算
                for i in range(input_data.shape[0]):
                    # 对每个通道进行计算
                    for j in range(input_data.shape[1]):
                        for k in range(self.weight.shape[0]):
                            # 计算本层的状态值
                            self.z[i, j * self.weight.shape[0] + k] = ConvolutionFunction.three_dem_convolution(self.weight[k], self.step, self.padding, input_data[i, j]) + self.bias[k]
                            # 计算本层的激活值
                            self.a[i, j * self.weight.shape[0] + k] = self.act_fun[0](self.z[i, j * self.weight.shape[0] + k])


        # 上一层为全连接层
        elif 'fc' in pre_layer_name:
            print('未提供此结构的前向计算公式：fc-conv')
            exit(1)

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
            # 初始化delta张量的大小
            self.delta = np.zeros(self.z.shape)
            # 计算所有训练集的delta

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
            #        self.delta[i, j] = task[i][j].result() * self.act_fun[1](self.a[i, j])
            #pool.shutdown(wait = True)


            # 串行计算
            # 对每个样本进行计算
            for i in range(self.delta.shape[0]):
                # 对每个通道进行计算
                for j in range(self.delta.shape[1]):
                    self.delta[i, j] = PoolFunction.three_dem_upsample(next_layer_w, next_layer_step, next_layer_delta[i, j], next_layer_pool_position[i, j].astype(int))* self.act_fun[1](self.a[i, j])


        # 下一层为卷积层
        elif 'conv' in next_layer_name:

            # 一对一连接
            if self.name[-2] == next_layer_name[-2]:
                # 初始化delta张量的大小
                self.delta = np.zeros(self.z.shape)
                # 将2，3维度的数据旋转180度
                nex_layer_w_r180 = np.rot90(next_layer_w, 2, (2, 3))

                # 多核并行计算
                #task = [[0 for t in range(self.delta.shape[1])]for i in range(self.delta.shape[0])]
                #pool = ProcessPoolExecutor(max_workers = cpu_count())
                ## 对每个样本进行计算
                #for i in range(self.delta.shape[0]):
                #    # 对每个通道进行计算
                #    for j in range(self.delta.shape[1]):
                #        # 放入进程池
                #        task[i][j] = pool.submit(ConvolutionFunction.three_dem_convolution, nex_layer_w_r180[j % nex_layer_w_r180.shape[0]], next_layer_step, next_layer_padding, next_layer_delta[i, j])
                ## 对每个样本进行计算
                #for i in range(self.delta.shape[0]):
                #    # 对每个通道进行计算
                #    for j in range(self.delta.shape[1]):
                #        self.delta[i, j] = task[i][j].result() * self.act_fun[1](self.a[i, j])
                #pool.shutdown(wait = True)


                # 串行计算
                # 对每个样本进行计算
                for i in range(self.delta.shape[0]):
                    # 对每个通道进行计算
                    for j in range(self.delta.shape[1]):
                        self.delta[i, j] = ConvolutionFunction.three_dem_convolution(nex_layer_w_r180[j % nex_layer_w_r180.shape[0]], next_layer_step, next_layer_padding, next_layer_delta[i, j]) * self.act_fun[1](self.a[i, j])


            # 一对多连接
            else:
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
                #            self.delta[i, j] += task[i][j][k].result() * self.act_fun[1](self.a[i, j])
                #pool.shutdown(wait = True)


                # 串行计算
                # 对每个样本进行计算
                for i in range(self.delta.shape[0]):
                    # 对每个通道进行计算
                    for j in range(self.delta.shape[1]):
                        for k in range(nex_layer_w_r180.shape[0]):
                           self.delta[i, j] += ConvolutionFunction.three_dem_convolution(nex_layer_w_r180[k], next_layer_step, next_layer_padding, next_layer_delta[i, j * nex_layer_w_r180.shape[0] + k]) * self.act_fun[1](self.a[i, j])


        # 下一层为全连接层
        elif 'fc' in next_layer_name:

            # 初始化delta张量的大小
            self.delta = np.zeros(self.z.shape)

            # temp 为2维数据，第一个维度为样本的信息，第二个维度为样本编号
            temp = np.dot(next_layer_w, next_layer_delta)

            # 低维映射到高维
            self.delta = MappingFunction.unmapping(temp, self.delta.shape) * self.act_fun[1](self.a)

        # 异常
        else:
            print(next_layer_name + '名字格式错误')
            exit(1)

        return self.delta

    # 本层网络参数进行调整
    def adjust_par(self, pre_layer_a, pre_layer_name, mu):
        """
        :param mu: 学习速率
        :param pre_layer_name: 上一层名字
        :param pre_layer_a: 上一层的激活值
        :return:
        """

        # 保存更新前的权重
        self.weight_last = self.weight.copy()
        self.bias_last = self.bias.copy()

        # 前一层为池化层
        if 'pool' in pre_layer_name:

            # 多核并行计算
            #task = [[[0 for k in range(self.a.shape[1] // self.weight.shape[0])] for t in range(self.weight.shape[0])]for i in range(pre_layer_a.shape[0])]
            #pool = ProcessPoolExecutor(max_workers = cpu_count())
            ## 对每个样本进行计算
            #for i in range(pre_layer_a.shape[0]):
            #    for j in range(self.weight.shape[0]):
            #        # 对每个通道进行计算
            #        for k in range(self.a.shape[1] // self.weight.shape[0]):
            #            # 调整w参数
            #            task[i][j][k] = pool.submit(ConvolutionFunction.three_dem_convolution, self.delta[i, k * self.weight.shape[0] + j], self.step, self.padding, pre_layer_a[i, k])
            #for i in range(pre_layer_a.shape[0]):
            #    for j in range(self.weight.shape[0]):
            #        # 对每个通道进行计算
            #        for k in range(self.a.shape[1] // self.weight.shape[0]):
            #            # 调整w参数
            #            self.weight[j] -= mu / pre_layer_a.shape[0] * task[i][j][k].result()
            #            # 调整b参数
            #            self.bias[j] -=  mu / pre_layer_a.shape[0] * np.sum(self.delta[i, k * self.weight.shape[0] + j])
            #pool.shutdown(wait = True)

            w_delta = np.zeros(self.weight.shape)
            b_delta = np.zeros(self.bias.shape)

            # 串行计算
            # 对每个样本进行计算
            for i in range(pre_layer_a.shape[0]):
                for j in range(self.weight.shape[0]):
                    # 对每个通道进行计算
                    for k in range(self.a.shape[1] // self.weight.shape[0]):
                        # 调整w参数
                        w_delta[j] += 1 / pre_layer_a.shape[0] * ConvolutionFunction.three_dem_convolution(self.delta[i, k * self.weight.shape[0] + j], self.step, self.padding, pre_layer_a[i, k])
                        # 调整b参数
                        b_delta[j] +=  1 / pre_layer_a.shape[0] * self.delta[i, k * self.weight.shape[0] + j].sum()

            w_delta = self.adam_w.cal_delta(w_delta)
            b_delta = self.adam_b.cal_delta(b_delta)

            self.weight -= mu * w_delta
            self.bias -= mu * b_delta

        # 前一层为卷积层
        elif 'conv' in pre_layer_name:

            # 一对一连接
            if self.name[-2] == pre_layer_name[-2]:

                # 多核并行计算
                #task = [[[0 for k in range(self.a.shape[1] // self.weight.shape[0])] for t in range(self.weight.shape[0])]for i in range(pre_layer_a.shape[0])]
                #pool = ProcessPoolExecutor(max_workers = cpu_count())
                ## 对每个样本进行计算
                #for i in range(pre_layer_a.shape[0]):
                #    for j in range(self.weight.shape[0]):
                #        # 对每个通道进行计算
                #        for k in range(self.a.shape[1] // self.weight.shape[0]):
                #            task[i][j][k] = pool.submit(ConvolutionFunction.three_dem_convolution, self.delta[i, k * self.weight.shape[0] + j], self.step, self.padding, pre_layer_a[i, k * self.weight.shape[0] + j])
                #for i in range(pre_layer_a.shape[0]):
                #    for j in range(self.weight.shape[0]):
                #        # 对每个通道进行计算
                #        for k in range(self.a.shape[1] // self.weight.shape[0]):
                #            # 调整w参数
                #            self.weight[j] -= mu / pre_layer_a.shape[0] * task[i][j][k].result()
                #            # 调整b参数
                #            self.bias[j] -=  mu / pre_layer_a.shape[0] * np.sum(self.delta[i, k * self.weight.shape[0] + j])
                #pool.shutdown(wait = True)

                w_delta = np.zeros(self.weight.shape)
                b_delta = np.zeros(self.bias.shape)

                # 串行计算
                # 对每个样本进行计算
                for i in range(pre_layer_a.shape[0]):
                    for j in range(self.weight.shape[0]):
                        # 对每个通道进行计算
                        for k in range(self.a.shape[1] // self.weight.shape[0]):
                            # 调整w参数
                            w_delta[j] += 1 / pre_layer_a.shape[0] * ConvolutionFunction.three_dem_convolution(self.delta[i, k * self.weight.shape[0] + j], self.step, self.padding, pre_layer_a[i, k * self.weight.shape[0] + j])
                            # 调整b参数
                            b_delta[j] +=  1 / pre_layer_a.shape[0] * self.delta[i, k * self.weight.shape[0] + j].sum()

                w_delta = self.adam_w.cal_delta(w_delta)
                b_delta = self.adam_b.cal_delta(b_delta)

                self.weight -= mu * w_delta
                self.bias -= mu * b_delta

            # 多对一连接
            else:

                # 多核并行计算
                #task = [[[0 for k in range(self.a.shape[1] // self.weight.shape[0])] for t in range(self.weight.shape[0])]for i in range(pre_layer_a.shape[0])]
                #pool = ProcessPoolExecutor(max_workers = cpu_count())
                ## 对每个样本进行计算
                #for i in range(pre_layer_a.shape[0]):
                #    for j in range(self.weight.shape[0]):
                #        # 对每个通道进行计算
                #        for k in range(self.a.shape[1] // self.weight.shape[0]):
                #            task[i][j][k] = pool.submit(ConvolutionFunction.three_dem_convolution, self.delta[i, k * self.weight.shape[0] + j], self.step, self.padding, pre_layer_a[i, k])
                #for i in range(pre_layer_a.shape[0]):
                #    for j in range(self.weight.shape[0]):
                #        # 对每个通道进行计算
                #        for k in range(self.a.shape[1] // self.weight.shape[0]):
                #            # 调整w参数
                #            self.weight[j] -= mu / pre_layer_a.shape[0] * task[i][j][k].result()
                #            # 调整b参数
                #            self.bias[j] -=  mu / pre_layer_a.shape[0] * np.sum(self.delta[i, k * self.weight.shape[0] + j])
                #pool.shutdown(wait = True)

                w_delta = np.zeros(self.weight.shape)
                b_delta = np.zeros(self.bias.shape)

                # 串行计算
                # 对每个样本进行计算
                for i in range(pre_layer_a.shape[0]):
                    for j in range(self.weight.shape[0]):
                        # 对每个通道进行计算
                        for k in range(self.a.shape[1] // self.weight.shape[0]):
                            # 调整w参数
                            w_delta[j] += 1 / pre_layer_a.shape[0] * ConvolutionFunction.three_dem_convolution(self.delta[i, k * self.weight.shape[0] + j], self.step, self.padding, pre_layer_a[i, k])
                            # 调整b参数
                            b_delta[j] +=  1 / pre_layer_a.shape[0] * self.delta[i, k * self.weight.shape[0] + j].sum()

                w_delta = self.adam_w.cal_delta(w_delta)
                b_delta = self.adam_b.cal_delta(b_delta)

                self.weight -= mu * w_delta
                self.bias -= mu * b_delta

        # 前一层为全连接层
        elif 'fc' in pre_layer_name:
            print('未提供此结构的权重调整公式：fc-conv')
            exit(1)

        # 异常
        else:
            print(pre_layer_name + '名字格式错误')
            exit(1)

    # 是否接受上次的参数调整
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
