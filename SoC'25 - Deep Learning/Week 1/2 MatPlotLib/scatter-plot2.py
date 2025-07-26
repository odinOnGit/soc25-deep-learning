import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
class1_x = np.random.uniform(-2, 2, 100)
class1_y = class1_x**2 + np.random.uniform(0.5, 3, 100)

class2_x = np.random.uniform(-2, 2, 100)
class2_y = class2_x**2 + np.random.uniform(0.5, 3, 100)

plt.scatter(class1_x, class1_y, color='blue', label='Class Blue')
plt.scatter(class2_x, class2_y, color='red', label='Class Red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot for Classification with Quadratic Boundary')
plt.grid(True)
plt.show()
