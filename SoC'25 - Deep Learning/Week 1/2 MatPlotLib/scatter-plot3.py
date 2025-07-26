import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
def cubic_function(x):
  return x**3 - 2*x + 5

class1_x = np.random.uniform(-3, 3, 150)
class1_y = cubic_function(class1_x) - np.random.uniform(1, 10, 150)

class2_x = np.random.uniform(-3, 3, 150)
class2_y = cubic_function(class2_x) + np.random.uniform(1, 10, 150)

plt.scatter(class1_x, class1_y, color='blue', label='Class Blue')
plt.scatter(class2_x, class2_y, color='red', label='Class Red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot for Classification with Cubic Boundary')
plt.grid(True)

plt.show()
