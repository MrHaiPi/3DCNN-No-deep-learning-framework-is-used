import time

import numpy as np

import DataIo
import NeuralNetwork

if __name__ == '__main__':

    # 神经网络参数保存目录（根）
    path_par_net_work = ""

    # 测试数据路径
    path_test_data = r"D:\Graduation project\数据集\已采用\SKIG\Sequence\dep\1_CV(1-2test)\test"

    # 测试数据
    test_data = (360, 1, 16, 64, 64)

    # 测试目标，一列为一个训练集的目标
    test_target = (10, 360)

    # 加载数据
    test_data, test_target = DataIo.load_data(path_test_data, test_data)

    # 加载保存的网络
    net_work = NeuralNetwork.NeuralNetwork()
    net_work.load_par(path_par_net_work)

    print("测试开始！")
    start = time.time()

    # 神经网络进行思考
    result = net_work.think(test_data)

    end = time.time()

    print("测试完成！耗时：", end - start,"s")

    print("神经网络输出为：")
    print(result.round(3))
    print("期望输出为：")
    print(test_target)

    temp = result[test_target == 1]
    temp1 = result.max(0) == temp
    rate = temp1[temp1 == True].size / test_target.shape[1]
    print("正确率为：", rate * 100, "%")
