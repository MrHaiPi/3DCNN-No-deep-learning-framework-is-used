import numpy as np
from numba import jit

@jit()
# 3维卷积
def three_dem_convolution(kernel, step, padding, input_data):
    '''
    :param padding: 补零个数（一半），内容为：[时间域补零个数，空间域高度补零个数，空间域宽度补零个数]
    :param step: 3D卷积步长，内容为：[时间域步长，空间域高度步长，空间域宽度步长]
    :param kernel: 3D卷积核，结构为：kernel[时间域编号][高度（像素）][宽度（像素）]
    :param input_data: 待3D卷积的数据，结构为：input_data[帧数][高度（像素）][宽度（像素）]
    :return: 特征立方体，结构为result[帧数][高度（像素）][宽度（像素）]，其中
    帧数 = (input_data.shape[0] + padding[0] * 2 - kernel.shape[0])/step[0] + 1
    高度（像素） = (input_data.shape[1]+ padding[1] * 2  - kernel.shape[1])/step[1] + 1
    宽度（像素） = (input_data.shape[2]+ padding[2] * 2  - kernel.shape[2])/step[2] + 1
    '''
    # 防止数据与卷积核、步长不匹配
    if input_data.shape[0] + padding[0] * 2 < kernel.shape[0] or \
            (input_data.shape[0] + padding[0] * 2 - kernel.shape[0]) % step[0] != 0:
        print("数据时间维度与卷积核、步长不匹配!")

    # 初始化计算结果
    result = np.zeros(((input_data.shape[0] + padding[0] * 2 - kernel.shape[0]) // step[0] + 1,
                       (input_data.shape[1] + padding[1] * 2 - kernel.shape[1]) // step[1] + 1,
                       (input_data.shape[2] + padding[2] * 2 - kernel.shape[2]) // step[2] + 1))
    for i in range(0,result.shape[0]):
        for j in range(0, kernel.shape[0]):
            # 时间域的padding
            if i + j - padding[0] >= input_data.shape[0] or i + j - padding[0] < 0:
                continue
            else:
                result[i] += two_dem_convolution(kernel[j], step[1:], padding[1:], input_data[i * step[0] + j - padding[0]])

    return result

@jit()
# 2维卷积
def two_dem_convolution(kernel, step, padding, input_data):
    '''
    :param padding: 补零个数（一半），内容为：[空间域高度补零个数，空间域宽度补零个数]
    :param step: 2D卷积步长，内容为：[空间域高度步长，空间域宽度步长]
    :param kernel: 2D卷积核，结构为：kernel[高度（像素）][宽度（像素）]
    :param input_data: 待2D卷积的数据，结构为：input_data[高度（像素）][宽度（像素）]
    :return: 特征图，结构为result[高度（像素）][宽度（像素）]，其中
    高度（像素） = (input_data.shape[0]+ padding[0] * 2 - kernel.shape[0])/step[0] + 1
    宽度（像素） = (input_data.shape[1]+ padding[0] * 2 - kernel.shape[1])/step[1] + 1
    '''
    # 防止数据与卷积核、步长不匹配
    if input_data.shape[0] + padding[0] * 2 < kernel.shape[0] or \
            (input_data.shape[0] + 2 * padding[0] - kernel.shape[0]) % step[0] != 0:
        print("数据空间维度高度与卷积核、步长不匹配!")

    # 初始化计算结果
    result = np.zeros(((input_data.shape[0] + padding[0] * 2 - kernel.shape[0]) // step[0] + 1,
                       (input_data.shape[1] + padding[1] * 2 - kernel.shape[1]) // step[1] + 1))
    for i in range(0,result.shape[0]):
        for j in range(0, kernel.shape[0]):
            # 时间域的padding
            if i + j - padding[0] >= input_data.shape[0] or i + j - padding[0] < 0:
                continue
            else:
                result[i] += one_dem_convolution(kernel[j], step[1:], padding[1:], input_data[i * step[0] + j - padding[0]])

    return result

@jit()
# 1维卷积
def one_dem_convolution(kernel, step, padding, input_data):
    '''
    :param padding: 补零个数（一半），内容为：[空间域宽度补零个数]
    :param step: 1D卷积步长，内容为：[空间域宽度步长]
    :param kernel: 1D卷积核，结构为：kernel[宽度（像素）]
    :param input_data: 待1D卷积的数据，结构为：input_data[宽度（像素）]
    :return: 特征图，结构为result[宽度（像素）]，其中
    宽度（像素） = (input_data.shape[0] + padding[0] * 2 - kernel.shape[0])/step[0] + 1
    '''
    # 防止数据与卷积核、步长不匹配
    if input_data.shape[0] + padding[0] * 2 < kernel.shape[0] or \
            (input_data.shape[0] + padding[0] * 2 - kernel.shape[0]) % step[0] != 0:
        print("数据空间维度宽度与卷积核、步长不匹配!")

    # 初始化计算结果
    result = np.zeros((input_data.shape[0] + padding[0] * 2 - kernel.shape[0]) // step[0] + 1)
    for i in range(0,result.shape[0]):
       for j in range(0, kernel.shape[0]):
           # 时间域的padding
           if i + j - padding[0] >= input_data.shape[0] or i + j - padding[0] < 0:
               continue
           else:
               result[i] += kernel[j] * input_data[i * step[0] + j - padding[0]]

    return result
