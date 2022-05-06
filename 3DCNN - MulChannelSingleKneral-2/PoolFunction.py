import numpy as np
from numba import jit

@jit()
# 3D池化操作
def three_dem_max_pool(kernel, step, input_data):
    '''
    :param kernel:池化核，结构为：kernel[时间域编号][高度（像素）][宽度（像素）]
    :param step:池化步长，内容为：[时间域步长，空间域高度步长，空间域宽度步长]
    :param input_data:待池化的数据，结构为：input_data[帧数][高度（像素）][宽度（像素）]
    :return: 两个特征立方体，一个为池化后的特征立方体，一个为池化位置的立方体，
    第一个结构为[帧数][高度（像素）][宽度（像素）]，帧数 = (input_data.shape[0] - kernel.shape[0])/step[0] + 1，高度（像素） = (input_data.shape[1] -  kernel.shape[1])/step[1] + 1，宽度（像素） = (input_data.shape[2] - kernel.shape[2])/step[2] + 1
    第二个结构与第一个结构相同
    '''
    # 防止数据与池化核、步长不匹配
    if  input_data.shape[0] < kernel.shape[0] or \
        input_data.shape[1] < kernel.shape[1] or \
        input_data.shape[2] < kernel.shape[2] or \
        (input_data.shape[0] - kernel.shape[0]) % step[0] != 0 or \
        (input_data.shape[1] - kernel.shape[1]) % step[1] != 0 or \
        (input_data.shape[2] - kernel.shape[2]) % step[2] != 0 :
        print("数据空间维度与池化核、步长不匹配!")

    # 初始化最大值和对应位置张量的大小
    result_1 = np.zeros(((input_data.shape[0] - kernel.shape[0]) // step[0] + 1,
                        (input_data.shape[1] - kernel.shape[1]) // step[1] + 1,
                        (input_data.shape[2] - kernel.shape[2]) // step[2] + 1))
    result_2 = np.zeros(result_1.shape)
    for i in range(result_1.shape[0]):
        for j in range(result_1.shape[1]):
            for k in range(result_1.shape[2]):
                # 拷贝池化核尺寸的立方体
                temp = input_data[i * step[0]:(i + 1) * step[0],j * step[1]:(j + 1) * step[1],k * step[2]:(k + 1) * step[2]]
                # 保存每个立方体的最大值
                result_1[i,j,k] = temp.max()
                # 保存每个立方体最大值的位置
                result_2[i,j,k] = temp.argmax()
    return result_1, result_2

@jit()
# 3D上采样操作
def three_dem_upsample(kernel, step, input_data, pool_position):
    '''
    :param pool_position: 池化位置，结构为[帧数][高度（像素）][宽度（像素）]，与input_data结构相同
    :param kernel:池化核，结构为：kernel[时间域编号][高度（像素）][宽度（像素）]
    :param step:池化步长，内容为：[时间域步长，空间域高度步长，空间域宽度步长]
    :param input_data:待上采样的数据，结构为：input_data[帧数][高度（像素）][宽度（像素）]
    :return: 特征立方体，结构为result[帧数][高度（像素）][宽度（像素）]，其中
    帧数 = (input_data.shape[0] - 1) * step[0] + kernel.shape[0]
    高度（像素） = (input_data.shape[1] - 1) * step[1] + kernel.shape[1]
    宽度（像素） = (input_data.shape[2] - 1) * step[2] + kernel.shape[2]
    '''
    # 防止数据与核、步长不匹配
    if  ((input_data.shape[0] - 1) * step[0]) % step[0] != 0 or \
        ((input_data.shape[1] - 1) * step[1]) % step[1] != 0 or \
        ((input_data.shape[2] - 1) * step[2]) % step[2] != 0 :
        print("数据空间维度与上采样核、步长不匹配!")

    # 初始化最大值和对应位置张量的大小
    result = np.zeros(((input_data.shape[0] - 1) * step[0] + kernel.shape[0],
                      (input_data.shape[1] - 1) * step[1] + kernel.shape[1],
                      (input_data.shape[2] - 1) * step[2] + kernel.shape[2]))
    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            for k in range(input_data.shape[2]):
                temp = result[i * step[0]:(i + 1) * step[0], j * step[1]:(j + 1) * step[1], k * step[2]:(k + 1) * step[2]]
                # 将输入的数据按照位置扩充
                #temp[np.unravel_index(pool_position[i,j,k], temp.shape)] = input_data[i,j,k]
                # 实现np.unravel_index
                pos = np.zeros(3)
                pos[0] = pool_position[i,j,k] // (temp.shape[1] * temp.shape[2])
                pos[1] = (pool_position[i,j,k] - pos[0] * (temp.shape[1] * temp.shape[2])) // temp.shape[2]
                pos[2] = pool_position[i,j,k] - pos[0] * (temp.shape[1] * temp.shape[2]) - pos[1] * temp.shape[2]
                temp[int(pos[0]), int(pos[1]), int(pos[2])] = input_data[i,j,k]

    return result
