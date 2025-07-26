import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def quadratic_function_1(x):
  return x**2 - 2

def quadratic_function_2(x):
  return -x**2 + 2

def quadratic_function_3(x):
  return x**2 + 2


class1_x = np.random.uniform(-2, 2, 100)
class1_y = quadratic_function_1(class1_x) - np.random.uniform(1, 3, 100)

class2_x = np.random.uniform(-2, 2, 100)
class2_y = np.random.uniform(quadratic_function_2(class2_x) + 0.5, quadratic_function_2(class2_x) - 0.5, 100)

class3_x = np.random.uniform(-2, 2, 100)
class3_y = quadratic_function_2(class3_x) + np.random.uniform(1, 3, 100)

plt.scatter(class1_x, class1_y, color='blue', label='Class 1')
plt.scatter(class2_x, class2_y, color='green', label='Class 2')
plt.scatter(class3_x, class3_y, color='red', label='Class 3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot for Multi-Class Classification with Quadratic Boundaries')
plt.legend
plt.grid(True)

x_curve = np.linspace(-2, 2, 400)
y_curve_1 = quadratic_function_1(x_curve)
y_curve_2 = quadratic_function_2(x_curve)
y_curve_3 = quadratic_function_3(x_curve)

plt.show()