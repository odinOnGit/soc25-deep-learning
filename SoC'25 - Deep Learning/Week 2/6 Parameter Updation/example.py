import numpy as np
def update_params(W, b, dW, db, learning_rate):
  W = W - learning_rate*dW
  b = b - learning_rate*db

  return W, b