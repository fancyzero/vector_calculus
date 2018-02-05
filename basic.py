import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def field_func(x,y):
    return np.array((np.sin(y),np.cos(x)))

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
            pp = field_func(vx, vy)
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

row = np.arange(-6,6,0.4)
column = np.arange(-6,6,0.4)
show_vector_field(field_func,row,column,"field")
show_scalar_field(divergence(field_func,row,column),"divergence")
show_scalar_field(curl(field_func,row,column),"curl")


