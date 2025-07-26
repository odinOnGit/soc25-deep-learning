import math

def basic_sigmoid(x):
  s = 1/(1 + math.exp(-x))
  return s

x = 2
print(basic_sigmoid(x))
