import time

import numpy as np

import DataIo
import NeuralNetwork

if __name__ == '__main__':

    # 神经网络参数保存目录（根）
    path_par_net_work = "stru4(99.8%train_83.9%test)"

    # 测试数据路径
    path_test_data = r"D:\Graduation project\数据集\已采用\SKIG\Sequence\rgb\1_CV(1test)\test"

    # 测试数据尺寸
    test_data_shape = (180, 3, 32, 64, 64)
    # 目标尺寸
    test_target_shape = (4, 180)
    # 加载测试数据
    test_data_path_list, test_target = DataIo.load_data_path(path_test_data, test_data_shape)

    # 加载保存的网络
    net_work = NeuralNetwork.NeuralNetwork()
    net_work.load_par(path_par_net_work)

    print("测试开始！")
    start = time.time()

    # 神经网络进行测试
    rate = net_work.test(test_data_path_list, test_target)

    end = time.time()

    print("测试完成！耗时：", end - start,"s")

    print("正确率为：", rate * 100, "%")
