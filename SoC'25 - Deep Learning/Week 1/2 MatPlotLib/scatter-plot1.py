import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class1_x = np.random.normal(2, 0.5, 50)
class1_y = np.random.normal(2, 0.5, 50)
class2_x = np.random.normal(4, 0.5, 50)
class2_y = np.random.normal(4, 0.5, 50)

plt.scatter(class1_x, class1_y, color='blue', label='Class 1')
plt.scatter(class2_x, class2_y, color='red', label='Class 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot for Classification')
plt.legend()
plt.grid(True)
plt.show()