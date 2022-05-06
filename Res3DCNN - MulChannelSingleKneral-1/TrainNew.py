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


    # 输入单个样本的结构
    size_input_data = list(train_data_shape[1:5])
    # 输出单个数据的结构
    size_output_data =  list((train_target_shape[0], 1))
    # 网络结构
    size_layer = ['conv1a','pool1','conv2a','pool2','conv3a','pool3','conv4a','pool4','conv5a','pool5','conv6a','pool6','fc1a','fc1b','fc1c']

    # 创造新的网络
    net_work = NeuralNetwork.NeuralNetwork()
    net_work.build_net(size_input_data, size_output_data, size_layer)

    # 开始训练网络
    net_work.train(train_data_path_list, train_target, test_data_path_list, test_target, batch_size = 10, epoch = 40, mu = 0.0001, path = path_par_net_work)
