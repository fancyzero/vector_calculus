import random
import numpy as np
import matplotlib.pyplot as plt

def field_func(x,y):
    return (np.sin(x),np.cos(y))

def show_vector_field(f):#f is a (x,x,2) array
    x = np.dsplit(f,2)
    u = np.squeeze(x[0])
    v = np.squeeze(x[1])
    fig, ax = plt.subplots()
    q = ax.quiver(u, v, units="dots")
    plt.show()

size = 40
dim = 2
f = np.ndarray((size,size,2))


for i in range(size):
    for j in range(size):
        f[i,j] = field_func(i/float(size)*3*np.pi,j/float(size)*3*np.pi)


show_vector_field(f)