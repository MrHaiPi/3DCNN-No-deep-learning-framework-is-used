import numpy as np
from numba import jit

@jit()
def mix(data_1, data_2, target_shape):
    result = np.zeros(target_shape)
    data = data_1 + data_2
    for j in range(target_shape[0]):
        max = 0
        index = 0
        for k in range(data.shape[1]):
            sum = data[j, k].sum()
            if sum > max:
                max = sum
                index = k

        for k in range(target_shape[1]):
            result[j, k] = data[j ,index]

    return result
