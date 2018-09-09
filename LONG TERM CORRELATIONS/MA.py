import numpy as np
import random as random
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
def ma_same_constants(ma, trials = 3, length = 1000):
    coef = 1
    # noise = np.random.normal(3,1,(trials, length))
    noise = np.random.uniform(0,1,(trials,length))
    arrays = np.zeros((trials, length, ma+1))
    arrays[:,:,0] = noise
    for i in range(1, ma+1):
        arrays[:,:,i] = np.roll(noise, i, axis=1)

    out = arrays[:,:,0] + np.sum(coef * arrays[:,:,1:], axis=2)
    out[:,:ma] = noise[:,:ma] #dont wrap around
    labels = np.ones((trials,1)) * ma
    return out, labels

def ma_random_coefficients(ma, trials, length):
    coeffs = np.random.uniform(0, 1, ma+1)
    coeffs[0] = 1

    noise = np.random.uniform(0, 1, (trials, length))
    arrays = np.zeros((trials, length, ma + 1))
    for i in range(0, ma + 1):
        arrays[:, :, i] = coeffs[i] * np.roll(noise, i, axis=1)
    out = np.sum(arrays, axis=2)
    out[:, :ma] = noise[:, :ma]
    labels = np.repeat(coeffs.reshape(1,-1), trials, axis=0)
    return out, labels

def labels_to_onehot(labels, N):
    result = np.pad(labels, (
        (0,0),
        (0,N - labels.shape[1])
    ), 'constant', constant_values=0)
    return result

def corr(x,y):
    numerator = (x - np.mean(x)) * (y-np.mean(y))
    return np.mean(numerator) / (np.std(x) * np.std(y))

def autocorr(x, n):
    result = []
    for i in range(n+1):
        result.append(corr(x, np.roll(x,i)))
    return result, list(range(n+1))


if __name__ == '__main__':
    a, l_a = ma_random_coefficients(2, 3, 10)
    b, l_b = ma_random_coefficients(3, 3, 10)
    ll_a = labels_to_onehot(l_a, 5)
    ll_b = labels_to_onehot(l_b, 5)
    out = np.concatenate((ll_a,ll_b), axis=0)
    print(out)


    # plt.plot(out)
    # plt.show()
