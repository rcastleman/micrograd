import math
import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.tanh(np.arange(-5,5,0.2)));plt.grid()
plt.plot(np.arange(-5,5,0.2)),np.tanh(np.arange(-5,5,0.2));plt.grid()

# from mod_import_test import add
# from vis import vis

from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    # dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

# print(add(7,3))
# print(vis(d))

# from lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb

def f(x):
    return 3*x**2 - 4*x + 5

xs = np.arange(-5,5,0.25)
ys = f(xs)
# print(xs)
# print(ys)
# plt.plot(xs,ys)
# plt.show()

# x = 2/3
# slope = 0 at 2/3
# h = 0.0001

def slope(x,h):
    return (f(x+h)-f(x))/h

# print(slope(x,h))

# let's get more complex
# a = 2.0
# b = -3.0
# c = 10.0
# def d(a,b,c): 
#     return a*b + c

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
    
    def __init__(self,data,_children=(),_op='',label = ''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        out = Value(self.data + other.data,(self,other),'+')
        return out
    
    def __mul__(self,other):    
        out = Value(self.data * other.data,(self,other),'*')
        return out
    
    def tanh(self):
       x = self.data
       t = (math.exp(2*x) - 1)/(math.exp(2*x)+1)
       out = Value(t, (self,),'tanh')
       return out

       

# a = Value(2.0, label = 'a')
# b = Value(-3.0, label = 'b')
# c = Value(10.0, label = 'c')
# e = a*b; e.label = 'e'
# d = e+c; d.label = 'd'
# f = Value(-2.0, label='f')
# L = d * f; L.label = 'L'

def lol():
  # h = 0.001
   
  # a = Value(2.0, label = 'a')
  # b = Value(-3.0, label = 'b')
  # c = Value(10.0, label = 'c')
  # e = a*b; e.label = 'e'
  # d = e+c; d.label = 'd'
  # f = Value(-2.0, label='f')
  # L = d * f; L.label = 'L';L.grad = 1.0
  # L1 = L.data

  delta = 0.01

  a = Value(2.0, label = 'a')
  a.grad = 6.0
  a.data += delta * a.grad
  # a.data += h
  b = Value(-3.0, label = 'b')
  b.grad = -4.0
  b.data += delta * b.grad
  # b.data += h
  c = Value(10.0, label = 'c')
  c.grad = -2.0
  c.data += delta * c.grad
  # c.data += h
  e = a*b; e.label = 'e'
  e.grad = -2.0
  # e.data += h
  d = e+c; d.label = 'd'
  d.grad = -2.0
  # d.data +=h
  f = Value(-2.0, label='f')
  f.grad = 4.0
  f.data += delta * f.grad
  L = d * f; L.label = 'L';L.grad = 1.0
  # L2 = L.data

  return(f"L.data = {L.data} where a.data = {a.data}, a.grad = {a.grad}, b.data = {b.data}, c.data = {c.data}, d.data = {d.data}, e.data = {e.data}, and f.data = {f.data}")

  # print("L1 =",L1)
  # print("L2 =",L2)
  # print(f"With h = {h} and d = {d.data}, (L2 - L1 / h) = ",((L2-L1)/h))

# print("d = a*b + c =",d)
# print(a*b+c) #python interprets "a+b" by using the __add__ or __mul__ function in the Class, where a = self and b = other

# print(draw_dot(L))
# print(d)
# print(d._prev)
# print(d._op)

# print(lol())

## -------Neurons--------
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.7, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
# o = n.tanh(); o.label = 'o'

print(f"x1 = {x1.data}, x2 = {x2.data}, w1 = {w1.data}, w2 ={w2.data}")
print(f"x1 * w1 = {x1w1.data}, x2 * w2 = {x2w2.data} and x1w1 + x2w2 = {x1w1x2w2.data}")
print(f"and b = {b.data} so n = {n.data}")

plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid()