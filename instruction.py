import math
import numpy as np
import matplotlib.pyplot as plt

# from lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb

def f(x):
    return 3*x**2 - 4*x + 5

xs = np.arange(-5,5,0.25)
ys = f(xs)
# print(xs)
# print(ys)
# plt.plot(xs,ys)
# plt.show()

x = 2/3
# slope = 0 at 2/3
h = 0.0001

def slope(x,h):
    return (f(x+h)-f(x))/h

# print(slope(x,h))

# let's get more complex
a = 2.0
b = -3.0
c = 10.0
def d(a,b,c): 
    return a*b + c

# #increase a by h to see sign of slope and magnitude of effect
# print("d^a = ",d(a,b,c))
# print("d1^a =",d(a+h,b,c))
# print("slope:",(d(a+h,b,c)-d(a,b,c))/h)

# #increase b by h to see sign of slope and magnitude of effect
# print("d^b = ",d(a,b,c))
# print("d1^b =",d(a,b+h,c))
# print("slope:",(d(a,b+h,c)-d(a,b,c))/h)

# #increase c by h to see sign of slope and magnitude of effect
# print("d^c = ",d(a,b,c))
# print("d1^c =",d(a,b,c+h))
# print("slope:",(d(a,b,c+h)-d(a,b,c))/h)

class Value:
    
    def __init__(self,data,_children=(),_op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        out = Value(self.data + other.data,(self,other),'+')
        return out
    
    def __mul__(self,other):    
        out = Value(self.data * other.data,(self,other),'*')
        return out

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c
e = b*c
print("d = a*b + c =",d)
# print(a*b+c) #python interprets "a+b" by using the __add__ or __mul__ function in the Class, where a = self and b = other
print("previous=",d._prev)
print("op = ",d._op)
print("e value:",e)
print("e children:",e._prev)
print("e operation:",e._op)
