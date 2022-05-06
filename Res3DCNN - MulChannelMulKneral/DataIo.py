import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
from skimage import io,transform
import numpy as np
import os

# 拉伸图片
def picture_resize(f, channel, height, width):
    rgb = io.imread(f)

    dst = transform.resize(rgb, (height, width))

    if channel == 1:
        temp = np.zeros((dst.shape[0], dst.shape[1], 1))
        temp[:,:,0] = dst[:,:,0]
        return temp
    else:
        return dst

# 旋转数据
def picture_rot(data, degree):
    result = np.rot90(data, degree / 90, (3, 4))
    return result

# 数据增强
def data_enhancement(data, label):
    # 数据增强（旋转）
    data_en = np.zeros((data.shape[0] * 2,) + data.shape[1:data.shape.__len__()])

    data_en[0:data.shape[0]] = data.copy()
    data_en[data.shape[0]:2 * data.shape[0]] = picture_rot(data, 180)

    label_en = np.tile(label, 2)

    return data_en, label_en

# 加载图片序列
def load_seq(path, shape_return):
    '''
    :param shape_return: 期望返回的shape[通道数][帧数][行][列]
    :return: 图片序列
    '''
    fil_list = os.listdir(path)
    # 排序
    fil_list.sort(key = lambda x:np.fromstring(x[0:-5], dtype=np.uint8).sum())

    step = fil_list.__len__() // shape_return[1]

    # 若样本的帧数少于指定的帧数，则用最后一帧复制补全，若多余指定的帧数，则按一定步长进行采样
    if step == 0:
        step = 1
        end = shape_return[1]
    else:
        end = fil_list.__len__()

    seq = np.zeros((shape_return[1], shape_return[2], shape_return[3], shape_return[0]))

    for k in range(0, end, step):

        index = k // step

        if index >= shape_return[1]:
            break

        seq[index] = picture_resize(path + '/' + fil_list[k], shape_return[0], shape_return[2], shape_return[3])

    # 调换矩阵的维度以符合数据的要求
    temp = np.swapaxes(seq, 0, 3)
    temp2 = np.swapaxes(temp, 1, 3)
    temp3 = np.swapaxes(temp2, 2, 3)

    return temp3

# 一次性加载数据
def load_data(path, shape_return):
    '''
    :param shape_return: 期望返回的shape[样本编号][通道数][帧数][行][列]
    :param path:
    :return: 数据和对应的标签
    '''

    # 获取指定目录下所有文件名
    dir_list = os.listdir(path)

    # 数据
    data = np.zeros(shape_return)

    # 标签
    label = np.zeros((dir_list.__len__(), shape_return[0]))

    # 样本类别
    for i in range(dir_list.__len__()):

        dir_list_2 = os.listdir(path + '/' + dir_list[i])

        # 某类别的样本数量
        for j in range(0, dir_list_2.__len__(), dir_list_2.__len__() // (shape_return[0] // dir_list.__len__())):

            seq = load_seq(path + '/' + dir_list[i] + '/' + dir_list_2[j], shape_return[2:5])

            data[i * (shape_return[0] // dir_list.__len__()) + j] = seq
            label[i, i * (shape_return[0] // dir_list.__len__()) + j] = 1

    return data, label

# 加载数据路径
def load_data_path(path, shape_return):
    '''
    :param shape_return: 期望返回的shape[样本编号][通道数][帧数][行][列]
    :return: 数据路径和对应的标签
    '''
    # 获取指定目录下所有文件名
    # 手势类别
    dir_list = os.listdir(path)
    dir_list.sort(key = lambda x:np.fromstring(x, dtype=np.uint8).sum())

    # 数据
    data_path = []

    # 标签
    label = np.zeros((dir_list.__len__(), shape_return[0]))

    for i in range(dir_list.__len__()):

        # 每个类别的总序列
        dir_list_2 = os.listdir(path + '/' + dir_list[i])

        for j in range(dir_list_2.__len__()):

            data_path.append(path + '/' + dir_list[i] + '/' + dir_list_2[j])

            label[i, data_path.__len__() - 1] = 1

    return data_path, label

# 根据序列名直接加载数据
def load_data_2(path_list, shape_return):
    '''
    :param shape_return: 期望返回的shape[通道数][帧数][行][列]
    :return: 数据和对应的标签
    '''
    data = np.zeros((path_list.__len__(),) + shape_return)

    for i in range(path_list.__len__()):

        data[i] = load_seq(path_list[i], shape_return)

    return data




if __name__ == "__main__":

    path = 'TrainData/Clic/Seq1/*.pnm'

    coll = io.ImageCollection(path, load_func = picture_resize)

    for i in range(len(coll)):
        io.imsave('TrainData/Clic/Seq1Resize/' + np.str(i) +'.pnm', np.array(coll[i]))  #循环保存图片
