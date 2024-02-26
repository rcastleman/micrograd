import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5

xs = np.arange(-5,5,0.25)
ys = f(xs)
# print(xs)
# print(ys)
# plt.plot(xs,ys)
# plt.show()

def slope(x,h):
    return (f(x+h)-f(x))/h

print(slope(3.0,0.00000001))
