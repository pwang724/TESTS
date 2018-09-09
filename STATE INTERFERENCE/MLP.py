from Data.ActivationF import *
from VISUALIZATION import netviz1
import numpy as np
import OdorState
import matplotlib.pyplot as plt

f = lambda x : relu(x)
df = lambda x : drelu(x)
output_f = lambda x : sigmoid(x)

# initialization
def init(dim_input, hidden_layers, dim_output):
    initialization_scale = .5
    looper = [dim_input] + hidden_layers + [dim_output]
    w, b = [], []
    for i in range(len(looper)-1):
        if i == 1:
            weight =  np.abs(np.random.randn(looper[i],looper[i+1]) * initialization_scale)
        else:
            weight = np.random.randn(looper[i], looper[i + 1]) * initialization_scale
        bias = 0 * np.random.randn(1, looper[i+1])
        w.append(weight)
        b.append(bias)
    return w, b

# make batches
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

def ff(w, b, XB):
    a = [None] * (len(hidden_layers) + 2)
    a[0] = XB
    for j in range(len(hidden_layers)):
        a[j + 1] = f(np.dot(a[j], w[j]) + b[j])
    a[-1] = output_f(np.dot(a[-2], w[-1]) + b[-1])
    return a

def bp(w, a, YB):
    err = [None] * (len(hidden_layers) + 2)
    dw = [None] * (len(hidden_layers) + 1)
    db = [None] * (len(hidden_layers) + 1)
    #errs
    err[-1] = a[-1] - YB
    for j in reversed(range(len(err) - 1)):
        err[j] = np.dot(err[j + 1], w[j].T) * df(a[j])
    #gradients
    for j in range(l):
        dw[j] = np.dot(a[j].T, err[j + 1])
        db[j] = np.sum(err[j+1], axis=0, keepdims=True)
    return dw, db, err

def loss(w, b, XB, YB):
    a = ff(w,b, XB)
    yhat = a[-1]
    yhat[yhat < 1e-10] = 1e-10
    with np.errstate(divide='raise'):
        try:
            l = - YB * np.log(yhat)  # this gets caught and handled as an exception
        except FloatingPointError:
            print('oh no!')

    return np.sum(l)

#parameters
env = OdorState.Env('blah')
hidden_layers = [3]
batch_size = 100
step_size = .001
N = 1000
X, Y = env.generate(1000)
n_samples, dim_input = X.shape
_, dim_output = Y.shape
w, b = init(dim_input, hidden_layers, dim_output)


#training
l = len(hidden_layers) + 1
curloss = 0
for e in range(5000):
    X, Y = env.generate(1000)
    XB, YB = make_batches(X, Y, batch_size)
    for i in range(len(XB)):
        a = ff(w, b, XB[i])
        dw, db, err = bp(w, a, YB[i])

        for j in range(l):
            w[j] += -step_size * dw[j]
            b[j] += -step_size * db[j]

        curloss += (- YB[i] * np.log(a[-1]) - (1 - YB[i]) * np.log(1 - a[-1])).sum()
    if e % 200 == 0 and e > 0:
        print('epoch: %3d err: %.2f' % (e, curloss))

    if curloss < 1:
        break
    curloss = 0

#weight viz
network = netviz1.NeuralNetwork()
for layer in range(len(a)):
    nNeuronsInLayer = a[layer].shape[1]
    if layer == len(a) - 1:
        bias = np.round(b[layer-1][0],2)
        network.add_layer(nNeuronsInLayer, bias=bias)
    elif layer == 0:
        network.add_layer(nNeuronsInLayer, w[layer].transpose())
    else:
        bias = np.round(b[layer-1][0],2)
        network.add_layer(nNeuronsInLayer, w[layer].transpose(), bias=bias)
network.draw()

#plot
N = 200
nr = env.N_ODORS
nc = env.N_STATES

for layer in range(1,3):
    nNeuron = a[layer].shape[1]
    for neuron in range(nNeuron):
        fig, ax = plt.subplots(nrows=nr, ncols=nc)
        nFigs = nr * nc
        for odor in range(3):
            for state in range(3):
                for toggle in range(3):
                    X, Y = env.generate_specific(N, odor, toggle)
                    a = ff(w, b, X)
                    out = a[layer][:, neuron]
                    relevant_state = X[:, 3 + state]

                    plt.sca(ax[odor][state])
                    plt.scatter(relevant_state, out)
                plt.legend(('toggle 1', 'toggle 2', 'toggle 3'))
#
plt.show()

    # #gradient checking
    # epsilon = 1e-4
    # for z in range(l):
    #     for x in range(w[z].shape[0]):
    #         for y in range(w[z].shape[1]):
    #             #ff
    #             x_ = XB[i][0,:].reshape(1,-1)
    #             y_ = YB[i][0,:].reshape(1,-1)
    #             a = ff(w, b, x_)
    #             dw, db, err = bp(w, a, y_)
    #             analyticGrad = dw[z][x,y]
    #
    #             # +epsilon
    #             w[z][x,y] += epsilon
    #             a = ff(w, b, x_)
    #             lossp = - y_ * np.log(a[l])
    #
    #             # -epsilon
    #             w[z][x,y] -= 2*epsilon
    #             a = ff(w, b, x_)
    #             lossm = - y_ * np.log(a[l])
    #
    #             numericalGrad = (lossp.sum() - lossm.sum()) / (2 * epsilon)
    #             diff = np.abs(numericalGrad - analyticGrad) / epsilon
    #             print(diff)
    #
    #             w[z][x,y] += epsilon









