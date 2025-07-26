import numpy as np

def calculate_loss(A, Y):
  m = Y.shape[0]
  cost = -(1/m)*np.sum(Y*np.log(A+1e-8) + (1-Y)*np.log(1-A + 1e-8))

  cost = np.squeeze(cost)

  return cost
