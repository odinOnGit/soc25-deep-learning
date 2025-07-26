import numpy as np

def backward_prop(A, X, Y):
  m = X.shape[1]
  dZ = A - Y
  dW = (1 / m)*np.dot(X, dZ.T)
  db = (1 / m)*np.sum(dZ)

  return dW, db
