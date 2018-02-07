import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import functools

def scalar_field_func(x,y):
    return np.sin(x)-np.cos(y)

def gradient(func, x,y,delta=0.00001):
    f = func(x,y)
    partial_x = func(x+delta,y)-f
    partial_y = func(x,y+delta)-f
    return np.array((partial_x/delta, partial_y/delta))

def partial_derivative(func, x,y,respectto, delta=(0.00001)):
    if (respectto == "x"):
        return ((func(x=x + delta, y=y) - func(x=x, y=y)) / delta)
    if (respectto == "y" ):
        return ((func(x=x, y=y+delta) - func(x=x, y=y)) / delta)


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
            f[i, j] = func(x=vx, y=vy)
    r = np.dsplit(f,2)
    u = np.squeeze(r[0]).T
    v = np.squeeze(r[1]).T
    plt.quiver(x, y, u, v, units="width")
    plt.title(title)
    plt.show()

def show_scalar_field(field,title=""):
    plt.imshow(field,origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.show()

row = np.arange(-3.14,3.14,0.2)
column = np.arange(-3.14,3.14,0.2)
func = functools.partial(gradient,func=scalar_field_func)
show_vector_field(func,row,column,"field")
show_scalar_field(divergence(func,row,column),"Laplacian") #Laplacian ( divergence of gradient of a scalar field )
show_scalar_field(curl(func,row,column),"curl") # curl of gradient it should be 0


