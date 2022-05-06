import DataIo
import NeuralNetwork
import numpy as np


if __name__== '__main__':

    # 神经网络参数保存目录（根）
    path_par_net_work = ""

    # 训练数据路径
    path_train_data = r"D:\Graduation project\数据集\已采用\SKIG\Sequence\dep\1_CV(1-2test)\train"

    # 测试数据路径
    path_test_data = r"D:\Graduation project\数据集\已采用\SKIG\Sequence\dep\1_CV(1-2test)\test"

    # 训练数据尺寸
    train_data_shape = (720, 1, 32, 64, 64)
    # 目标尺寸
    train_target_shape = (10, 720)
    # 加载数据
    train_data_path_list, train_target = DataIo.load_data_path(path_train_data, train_data_shape)

    # 测试数据尺寸
    test_data_shape = (360, 1, 32, 64, 64)
    # 目标尺寸
    test_target_shape = (10, 360)
    # 加载测试数据
    test_data_path_list, test_target = DataIo.load_data_path(path_test_data, test_data_shape)

    # 创造网络
    net_work = NeuralNetwork.NeuralNetwork()

    print("加载参数文件(" + path_par_net_work + ")...")
    # 加载指定的参数
    net_work.load_par(path_par_net_work)

    # 开始训练网络
    net_work.train(train_data_path_list, train_target, test_data_path_list, test_target, batch_size = 10, epoch = 10, mu = 0.00001, path = path_par_net_work)





