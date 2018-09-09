import numpy as np


class Env(object):

    PAIRING = np.array([[0,1,2], [0,1,2]])
    SIZE = []
    N_ODORS = 0
    N_STATES = 0


    def __init__(self, pairing):
        self.SIZE = np.apply_along_axis(lambda x : len(x), 1, self.PAIRING)
        self.N_ODORS = self.SIZE[0]
        self.N_STATES = self.SIZE[1]

    def make_states(self, odors, states, toggles):
        X = np.concatenate((odors, states, toggles), axis = 1)
        toggleIdx = np.where(toggles == 1)[1]
        permuted_states = states[:, self.PAIRING[1,:]]
        y = odors * permuted_states
        Y = y[range(len(toggleIdx)), toggleIdx].reshape(-1,1)
        return X, Y


    def generate(self, N):
        odors = np.zeros((N, self.N_ODORS))
        odorIdx = np.random.randint(self.N_ODORS, size = N)
        odors[range(len(odorIdx)), odorIdx] = 1

        states = np.random.rand(N, self.N_STATES)
        toggleIdx = np.random.randint(self.N_STATES, size = N)
        toggles = np.zeros_like(states)
        toggles[range(len(toggleIdx)),toggleIdx] = 1

        X, Y = self.make_states(odors, states, toggles)
        return X, Y

    def generate_specific(self, N, odor, toggle):
        odors = np.zeros((N, self.N_ODORS))
        odors[:,odor] = 1
        states = np.random.rand(N, self.N_STATES)
        toggles = np.zeros((N, self.N_STATES))
        toggles[:, toggle] = 1

        X, Y = self.make_states(odors, states, toggles)
        return X, Y



if __name__ == "__main__":
    a = Env('blah')
    X, Y = a.generate(100)
    Xs, Ys = a.generate_specific(20, 0, 0)
    print(Xs)
    print(Ys)