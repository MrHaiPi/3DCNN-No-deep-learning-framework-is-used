import numpy as np

# 选择一个激活函数
def act_fun(name):
    if name == 'sigmoid':
        return [sigmoid, sigmoid_derivative]
    elif name == 'relu':
        return [relu, relu_derivative]
    elif name == 'leaky_relu':
        return [leaky_relu, leaky_relu_derivative]
    elif name == 'softmax':
        return [softmax, softmax_derivative]
    elif name == 'tanh':
        return [tanh, tanh_derivative]

# sigmoid激活函数
def sigmoid(x):
    #applying the sigmoid function
    return 1 / (1 + np.exp(-x))
# sigmoid激活函数导数
def sigmoid_derivative(x):
    #computing derivative to the Sigmoid function
    temp = x * (1 - x)
    return temp

# 线性整流函数（Rectified Linear Unit, ReLU）
def relu(x):
    temp = x.copy()
    temp[ temp < 0] = 0
    return temp
# ReLU的导数
def relu_derivative(x):
    temp = x.copy()
    temp[temp <= 0] = 0
    temp[temp > 0] = 1
    return temp

# 线性整流函数Leaky ReLU
def leaky_relu(x):
    temp = x.copy()
    # x = max(0,x)，即将x中小于0的值设为0
    temp[ temp < 0] = 0.01 * temp[ temp < 0]
    return temp
# Leaky ReLU的导数
def leaky_relu_derivative(x):
    temp = x.copy()
    temp[temp <= 0] = 0.01
    temp[temp > 0] = 1
    return temp

# softmax函数
def softmax(x):
    # 防止数值溢出
    # 获取每个样本的最大值
    d = x.max(0)
    d = np.tile(d,(x.shape[0], 1))
    x_d = np.exp(x - d)
    x_d_sum = np.tile(x_d.sum(0), (x.shape[0], 1))
    return x_d / x_d_sum
# softmax函数的导数
def softmax_derivative(x):
    return np.ones(x.shape)

# tanh函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
# tanh函数的导数
def tanh_derivative(x):
    return 1 - x * x
