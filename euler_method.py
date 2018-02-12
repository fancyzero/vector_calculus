import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import functools

def func_1(y,step = 0.01):
    return y+y*step

def ff(initial, step = 0.01):
    y = []
    x = []
    y2 = []
    p = initial
    for i in range(int(10/step)):
        x.append(i*step)
        y2.append(np.exp(i*step))
        pp = func_1(p,step)
        p = pp
        y.append(pp)

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.show()

ff(1,0.1)
