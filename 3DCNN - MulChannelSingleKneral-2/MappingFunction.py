import numpy as np
from numba import jit

@jit()
# 映射(高维向低维映射)
def mapping(intput, target_shape):

    result = np.zeros(target_shape)

    for j in range(result.shape[1]):
        result[:, j] = intput[j].reshape(result[:, j].shape)

    return result

@jit(forceobj=True)
# 反映射（低维向高维映射）
def unmapping(intput, traget_shape):

    result = np.zeros(traget_shape)

    for i in range(result.shape[0]):
        result[i] = intput[:, i].reshape(result[i].shape)

    return result
