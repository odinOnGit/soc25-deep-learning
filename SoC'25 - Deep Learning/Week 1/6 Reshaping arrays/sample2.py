import numpy as np

arr = np.arange(1, 17)
print("Original Array: ", arr)
reshaped_arr = arr.reshape(4, 4)
print("Reshaped with arg(4, 4): ", reshaped_arr)

reshaped_arr2 = arr.reshape(4, -1)
print("Reshaped with arg(4, -1): ", reshaped_arr2)