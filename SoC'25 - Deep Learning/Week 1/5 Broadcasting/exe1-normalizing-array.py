import numpy as np

arr = np.array([[1, 2, 3], [6, 7, 8]])
def normalized_array(x):
  min = np.min(x)
  max = np.max(x)
  normArr = np.array(([arr] - min)/(max - min))

  return normArr

print(normalized_array(arr))