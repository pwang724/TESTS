import numpy as np

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    y = np.clip(y, -50, 50)
    return y * (1.0 - y)


def relu(x):
    return np.maximum(0, x)


def drelu(x):
    return x > 0


def tanh(x):
    x = np.clip(x, -50, 50)
    return np.tanh(x)


def dtanh(x):
    x = np.clip(x, -50, 50)
    return 1 - (x ** 2)