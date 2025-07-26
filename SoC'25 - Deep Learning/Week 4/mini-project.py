import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

X_train = train.drop("label", axis = 1).values / 225.0
Y_train = train["label"].values

X_test = test.values / 255.0

def one_hot(y, num_classes=10):
    m = y.shape[0]
    one_hot_matrix = np.zeros((m, num_classes))
    one_hot_matrix[np.arange(m), y] = 1
    return one_hot_matrix

Y_train = one_hot(Y_train)

def init_params(input_dim, num_classes):
    W = np.random.randn(input_dim, num_classes) * 0.01
    b = np.zeros((1, num_classes))
    return W, b

def softmax(Z):
    exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def accuracy(Y_pred, Y_true):
    pred_labels = np.argmax(Y_pred, axis=1)
    true_labels = np.argmax(Y_true, axis=1)
    return np.mean(pred_labels == true_labels)

def forward(X, W, b):
    Z = np.dot(X, W) + b
    return softmax(Z)

def compute_loss(Y_pred, Y_true):
    m = Y_true.shape[0]
    loss = -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m
    return loss

def backward(X, Y_pred, Y_true):
    m = X.shape[0]
    dZ = (Y_pred - Y_true) / m
    dW = np.dot(X.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)
    return dW, db

def train_model(X, Y, num_classes=10, lr=0.1, epochs=1000):
    n_features = X.shape[1]
    W, b = init_params(n_features, num_classes)

    for i in range(epochs):
        Y_pred = forward(X, W, b)
        loss = compute_loss(Y_pred, Y)
        acc = accuracy(Y_pred, Y)
        dW, db = backward(X, Y_pred, Y)

        W -= lr * dW
        b -= lr * db

        if i%100 ==0:
            print(f"Epoch {i} - Loss: {loss: .4f} - Accuracy: {acc*100:.2f}%")

    return W, b

W, b = train_model(X_train, Y_train, epochs=1000, lr=0.5)

Y_train_pred_probs = forward(X_train, W, b)
Y_train_preds = np.argmax(Y_train_pred_probs, axis = 1)

def predict(X, W, b):
    probs = forward(X, W, b)
    return np.argmax(probs, axis=1)

preds = predict(X_test, W, b)

submission = pd.DataFrame({"ImageId": np.arange(1, len(preds) + 1), "Label": preds})
submission.to_csv("submission.csv", index=False)

for i in range(10):
    index = np.random.randint(0, X_train.shape[0])
    image = X_train[index].reshape(28, 28)
    label = np.argmax(Y_train[index])
    prediction = Y_train_preds[index]

    plt.imshow(image, cmap='gray')
    plt.title(f"True: {label} | Predicted: {prediction}")
    plt.axis('off')
    plt.show()
