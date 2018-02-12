import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from perlinnoise import *

import functools
gdelta = 1.0/2**16
def scalar_field_func(x,y):
    return np.sin(x) - np.cos(y)

def gradient(func, x,y,delta=gdelta):
    f = func(x,y)
    partial_x = (func(x+delta,y)-f)/delta
    partial_y = (func(x,y+delta)-f)/delta
    return np.array((partial_x, partial_y))

def partial_derivative(func, x,y,respectto, delta=gdelta):
    if (respectto == "x"):
        px = ((func(x=x + delta, y=y) - func(x=x, y=y)) / delta)
        return px
    if (respectto == "y" ):
        py = ((func(x=x, y=y + delta) - func(x=x, y=y)) / delta)
        return py


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


if __name__ == "__main__":
    seed=init_perlin_seed(3,3)
    f = np.ndarray((128, 128))
    f2 = np.ndarray((128, 128))
    for i in range(128):
        for j in range(128):
            f2[i,j] = noise_scalar_field( i/128.0,j/128.0, seed)

    plt.imshow(f2,cmap="gray",origin='lower')
    plt.title("perlin noise")
    plt.show()



    row = np.arange(0,1,0.04)
    column = np.arange(0,1,0.04)

    perlinnoise_func = functools.partial(noise_scalar_field,seed = seed)
    func = functools.partial(gradient,func=perlinnoise_func)
    p = np.array(((92,48),(92,49),(92,50),(92,51),(92,52)))/100.0
    for pp in p:
        print(partial_derivative(func,pp[0],pp[1],"x")[0] , partial_derivative(func,pp[0],pp[1],"y")[1])
        print(partial_derivative(func, pp[0], pp[1], "x")[0]+ partial_derivative(func, pp[0], pp[1], "y")[1])

    show_vector_field(func,row,column,"field")
    div = divergence(func, row, column)
    show_scalar_field(div.T,"Laplacian") #Laplacian ( divergence of gradient of a scalar field )
    show_scalar_field(curl(func,row,column),"curl") # curl of gradient it should be 0


