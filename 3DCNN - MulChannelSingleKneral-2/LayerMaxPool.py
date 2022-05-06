import numpy as np
import ConvolutionFunction
import MappingFunction
import PoolFunction
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

class LayerMaxPool:
    # 最大池化层
    def __init__(self, size_of_kernel, step, name):
        """
        :param size_of_kernel: 核尺寸：[时间域长度，空间域高度，空间域宽度]
        :param step: 步长尺寸：[时间域步长，空间域高度步长，空间域宽度步长]
        :param name: 本层网络的名字
        """
        # 池化步长
        self.step = step

        # 本层网络的名字
        self.name = name

        # 本层网络的输入值（状态值）
        self.z = None

        # 本层网络的输出值（激活值）
        self.a = None

        # 本层网络的池化过程中取最大值的位置
        self.pool_position = None

        # 本层网络的delta
        self.delta = None

        # 池化核
        self.weight = np.ones(size_of_kernel)

    # 本层网络进行思考
    def think(self, input_data, pre_layer_name):
        """
        :param pre_layer_name: 前一层的名字
        :param input_data: 输入数据，为5维数组，结构为：input_data[样本编号][通道数][帧数][行][列]
        :return:本层网络的激活态
        """

        # 前一层为池化层或卷积层
        if 'pool' or 'conv'  in pre_layer_name:
            # 初始化状态张量和激活张量的大小
            self.z = np.zeros((input_data.shape[0],
                               input_data.shape[1],
                               (input_data.shape[2] - self.weight.shape[0]) // self.step[0] + 1,
                               (input_data.shape[3] - self.weight.shape[1]) // self.step[1] + 1,
                               (input_data.shape[4] - self.weight.shape[2]) // self.step[2] + 1))
            # 池化层的激活张量就是本层的状态张量pool操作后的值
            self.a = np.zeros(self.z.shape)

            # 初始化池化位置张量的大小，每个位置记录self.a中对应值在self.z中的位置
            self.pool_position = np.zeros(self.a.shape)


            # 串行计算
            # 计算最新的状态和激活值
            # 对每个样本进行计算
            for i in range(input_data.shape[0]):
                for j in range(input_data.shape[1]):
                    # 计算本层的状态值
                    result = PoolFunction.three_dem_max_pool(self.weight, self.step, input_data[i, j])
                    self.z[i, j] = result[0]
                    # 计算本层的激活值
                    self.a[i, j] = self.z[i, j]
                    # 保存池化位置
                    self.pool_position[i, j] = result[1]


        # 前一层为全连接层
        elif 'fc' in pre_layer_name:
            print('未提供此结构的前向计算公式：fc-pool')
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

            # 串行计算
            # 对每个样本进行计算
            for i in range(self.delta.shape[0]):
                for j in range(self.delta.shape[1]):
                    for k in range(nex_layer_w_r180.shape[0]):
                       self.delta[i, j] += ConvolutionFunction.three_dem_convolution(nex_layer_w_r180[k], next_layer_step, next_layer_padding, next_layer_delta[i, k])


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
