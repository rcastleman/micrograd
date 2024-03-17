import math
import numpy as np
import matplotlib.pyplot as plt

delta = 0.01

e = a * b 
d = e + c 
L = d * f 


a.data += delta * a.grad
b.data += delta * b.grad
c.data += delta * c.grad
f.data += delta * f.grad