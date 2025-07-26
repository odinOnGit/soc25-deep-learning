import numpy as np
import matplotlib.pyplot as plt
import os

image_path = "./images/cat.jpeg"

if not os.path.exists(image_path):
  print(f"Error: File '{image_path}' not found.")

else:
  image = plt.imread(image_path)
  plt.imshow(image)
  plt.axis('off')
  plt.show()