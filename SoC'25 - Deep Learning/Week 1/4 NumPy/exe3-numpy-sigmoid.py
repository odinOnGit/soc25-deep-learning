import numpy as np

def numpy_sigmoid(x):
  s = 1/(1+np.exp(-x))
  return s

x = 2
print(numpy_sigmoid(x))