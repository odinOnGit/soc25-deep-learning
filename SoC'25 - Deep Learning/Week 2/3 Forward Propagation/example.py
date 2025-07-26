import numpy as np

def forward_prop(W, b, X):
  Z = np.dot(W.T, X) + b
  A = 1/(1+np.exp(-Z))

  return A