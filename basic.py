import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from random import gauss
def random_unit_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]


def init_perlin_seed(w,h):
    seed={}
    seed["grid"] =  np.ndarray((w,h,2))

    i = np.nditer(seed,flags=["external_loop","buffered"],op_flags=["readwrite"],buffersize=2)
    while not i.finished:
        i[0]=random_unit_vector(2)
        i.iternext()
    seed["size"]=(w,h)
    return seed


def noise_scalar_field(perlin_seed,x,y):
    seed["size"]
    pass


def scalar_field_func(x,y):
    return np.sin(x) - np.cos(y)

def gradient(func,x,y,delta=(0.00001)):
    partialx= (func(x+delta,y) - func(x,y))/delta
    partialy= (func(x,y+delta) - func(x,y))/delta
    return np.array((partialx, partialy))

def partial_derivative(func, x,y,respectto, delta=(0.00001)):
    if (respectto == "x"):
        return ((func(x + delta, y) - func(x, y)) / delta)
    if (respectto == "y" ):
        return ((func(x, y+delta) - func(x, y)) / delta)


def divergence(func,x, y):
    div = np.ndarray((len(x),len(y)))
    for i,vx in enumerate(x):
        for j,vy in enumerate(y):
            div[i,j] = partial_derivative(func,vx,vy,"x")[0] + partial_derivative(func,vx,vy,"y")[1]
    return div

def curl(func, x, y):
    curl = np.ndarray((len(x),len(y)))
    for i,vx in enumerate(x):
        for j,vy in enumerate(y):
            curl[i,j] = partial_derivative(func,vx,vy,"x")[1] - partial_derivative(func,vx,vy,"y")[0]
    return curl

def show_vector_field(func,x,y,title=""):
    f=np.ndarray((len(x),len(y),2))
    for i,vx in enumerate(x):
        for j,vy in enumerate(y):
            pp = gradient(func,vx, vy)
            f[i, j] = pp
    r = np.dsplit(f,2)
    u = np.squeeze(r[1])
    v = np.squeeze(r[0])
    plt.quiver(x,y,u, v, units="width")
    plt.title(title)
    plt.show()

def show_scalar_field(field,title=""):
    plt.imshow(field,origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.show()


seed=init_perlin_seed(100,100)

for i in range(1024):
    for j in range(1024):
        f = np.ndarray((1024,1024))
        f[i,j] = noise_scalar_field(seed, i,j,1024,1024)


row = np.arange(-6,6,0.4)
column = np.arange(-6,6,0.4)
show_vector_field(scalar_field_func,row,column,"field")
show_scalar_field(divergence(scalar_field_func,row,column),"divergence")
show_scalar_field(curl(scalar_field_func,row,column),"curl")


