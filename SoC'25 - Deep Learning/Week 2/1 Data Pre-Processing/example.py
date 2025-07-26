import numpy as np
import matplotlib.pyplot as plt


dataset_raw = np.genfromtxt("C:/Users/ASUS/OneDrive/Desktop/SoC'25 - Deep Learning/Week 2/Data Pre-Processing/datasets/heart.csv", dtype="str", delimiter=",")
print(dataset_raw.shape)
headers = dataset_raw[0, :]
print(headers)

dataset = dataset_raw[1:, :]
dataset = dataset.astype(float)
print(dataset)

X = dataset[:, :13]
Y = dataset[:, 13]
print(X.shape)
print(Y.shape)

X = X.T
print(X.shape)
print(Y.shape)

index = int(0.8 * X.shape[1])

X_train = X[:, :index]
X_test = X[:, index:]

Y_train = Y[:index]
Y_test = Y[index:]

print("X_train shape", X_train.shape)
print("Y_train shape", Y_train.shape)
print("Number of training examples =", Y_train.shape[0])
print("-"*40)
print("X_test shape", X_test.shape)
print("Y_test shape", Y_test.shape)
print("Number of testing examples =", Y_test.shape[0])