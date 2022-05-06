import math
import numpy as np
from numba import jit

class Adam:
    def __init__(self, p1 = 0.9, p2 = 0.9995, sc = math.pow(0.1, 10), t = 0):

        # 指数衰减速率
        self.p1 = p1
        self.p2 = p2

        # 小常数
        self.sc = sc

        # 一阶矩变量
        self.s = None

        # 二阶矩变量
        self.r = None

        # 时间步长
        self.t = t

    @jit(forceobj=True)
    def cal_delta(self, delta):

        if self.s is None:
            self.s = np.zeros(delta.shape)
        if self.r is None:
            self.r = np.zeros(delta.shape)

        self.t += 1

        self.s = self.p1 * self.s + (1 - self.p1) * delta
        self.r = self.p2 * self.r + (1 - self.p2) * delta * delta

        s_temp = self.s / (1 - math.pow(self.p1, self.t))
        r_temp = self.r / (1 - math.pow(self.p2, self.t))

        #new_delta = np.zeros(delta.shape)
        #if delta.max() != 0 or delta.min() != 0:
        new_delta = s_temp / (np.sqrt(r_temp) + self.sc)

        return new_delta
