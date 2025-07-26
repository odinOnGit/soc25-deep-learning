import numpy as np

numpy_array = np.array([1, 2, 3, 4, 5])
def numpy_sigmoid(x):
  result_array = np.array(x)
  result_array = 1/(1+np.exp(-numpy_array))
  return result_array

print(numpy_sigmoid(numpy_array))

