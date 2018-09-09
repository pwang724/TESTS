import numpy as np

def make_batches(X, Y, batch_size):
    n_samples, dim_input = X.shape
    perm = np.random.permutation(n_samples)
    X = X[perm, :]
    Y = Y[perm, :]
    n_batches = n_samples // batch_size
    XB = []
    YB = []
    for j in range(n_batches):
        XB.append(X[j * batch_size:(j + 1) * batch_size, :])
        YB.append(Y[j * batch_size:(j + 1) * batch_size, :])
    return XB, YB