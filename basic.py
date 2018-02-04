import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def field_func(x,y):
    return np.array(np.sin(x),np.cos(y))

def derivative(func, x,y,partialx, partialy, delta=(0.0001)):
    dx = 0
    dy = 0
    if (partialx == 1):
        dx = delta
    if (partialy == 1 ):
        dy = delta
    return (func(x+dx,y+dy)-func(x,y))/delta

def divergence(func,x, y):
    div = np.ndarray((len(x),len(y)))
    for i,vx in enumerate(x):
        for j,vy in enumerate(y):
            div[i,j] = derivative(func,vx,vy,1,0) + derivative(func,vx,vy,0,1)
    return div

def show_vector_field(func,x,y):
    f=np.ndarray((len(x),len(y),2))
    for i,vx in enumerate(x):
        for j,vy in enumerate(y):
            f[i, j] = field_func(vx, vy)
    x = np.dsplit(f,2)
    u = np.squeeze(x[0])
    v = np.squeeze(x[1])
    fig, ax = plt.subplots()
    q = ax.quiver(u, v, units="dots")
    plt.show()

def show_scalar_field(field):
    plt.imshow(field)
    plt.show()


#show_vector_field(field_func,np.arange(0,np.pi*3, 0.1),np.arange(0,np.pi*3, 0.1))
show_scalar_field(divergence(field_func,np.arange(0,np.pi*3, 0.1),np.arange(0,np.pi*3, 0.1)))