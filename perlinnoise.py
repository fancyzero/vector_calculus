import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from random import gauss
import itertools
import random
random.seed(0)
def random_unit_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]

def init_perlin_seed(w,h):
    seed={}
    seed["grid"] =  np.ndarray((w,h,2))
    i = np.nditer(seed["grid"],flags=["external_loop","buffered"],op_flags=["writeonly"],buffersize=2)
    while not i.finished:
        i[0]=random_unit_vector(2)
        i.iternext()
    seed["size"]=(w,h)
    return seed

def interpolate(a,b,alpha):
    x = alpha
    alpha = -20*x**7+70*x**6 - 84*x**5 + 35*x**4
    return a+(b-a)*alpha

def gridgradient(p,q):
    p = np.array(p)
    q = np.array(q)
    return np.dot(p,q)

def noise_scalar_field(x,y,seed):
    size = np.array(seed["size"])
    size -= 1
    x = x * size[0]
    y = y * size[1]
    grid = seed["grid"]
    gc = np.array(list(itertools.product([0,1],[0,1] )))+np.array((int(x),int(y)))
    g0 = grid[tuple(gc[0])]
    g1 = grid[tuple(gc[1])]
    g2 = grid[tuple(gc[2])]
    g3 = grid[tuple(gc[3])]
    x00 = gridgradient(np.array((x, y))-gc[0], g0)
    x01 = gridgradient(np.array((x, y))-gc[1], g1)
    x10 = gridgradient(np.array((x, y))-gc[2], g2)
    x11 = gridgradient(np.array((x, y))-gc[3], g3)
    a = interpolate(x00,x01,y-gc[0][1])
    b = interpolate(x10,x11,y-gc[2][1])
    r = interpolate(a,b,x-gc[0][0])
    return r
