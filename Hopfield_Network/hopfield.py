import numpy as np

class Hopfield(object):
    """Hopfield Network"""

    def __init__(self, max_size):
        ''' Creates a multi pattern pattern hopfield network 

            max_size -- the maximum number of nodes in the network
        '''
        self.size = max_size
        self.weights = np.zeros((self.size, self.size))
        self._initialise_nodes()
        self.trained = False

    def _initialise_nodes(self):
        ''' Initialises the nodes randomly to -1 or 1 '''
        self.nodes = np.random.choice([-1, 1], self.size)

    def _train_weights(self, training_vectors):
        ''' trains all the weights using the input vector

            training_vectors -- a list of numpy training vectors
        '''
        for v in training_vectors:
            self.weights = np.add(self.weights, np.outer(v, v))
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(training_vectors)

    def _update_node(self, i):
        ''' Updates a node i with the weights, may flip the node values

            i -- ith node
        '''
        h_i = self._calulate_h_i(i)
        self.nodes[i] = self._updating_rule(h_i)

    def _calulate_h_i(self, i):
        ''' Calculates the valuf of h_i

            i -- ith node
        '''
        return np.dot(self.weights[i,], self.nodes)

    def _updating_rule(self, h_i):
        ''' Uses specified updating rule

            h_i -- the intermediate hebbian of node id
            Returns -- the value of the node
        '''
        return 1 if h_i > 0 else -1

    def _update(self, vector, time=50000, interval=0):
        ''' Updates the hopfield network

            vector -- the initial vector
            time -- the number of iterations to run for
        '''
        self.nodes = vector
        state = []
        for t in range(time):
            i = np.random.randint(0, self.size - 1)
            self._update_node(i)
            if interval != 0 and t % interval == 0:
                state.append(np.array(self.nodes))
        return state

    def calculate_energy(self):
        ''' Calculates and returns the energy of the network.

            returns -- E = -(1/2)sum(i,j) w_ij * n_i * n_j = -(1/2) N^T * W * N
        '''
        return - (1/2) * np.sum(self.nodes[:,None] * self.weights * self.nodes)

    def train(self, v):
        ''' Trains the network with an input vector '''
        self._train_weights(v)

    def test_with_random(self):
        ''' Tests the network with a random image '''
        test = np.random.choice([-1, 1], self.size)
        log = self._update(test, time=1000000, interval=0)
        log.append(self.nodes)
        return log


class StochasticHopfield(Hopfield):
    ''' Stochastic Hopfield Network '''

    def __init__(self, max_size, T):
        ''' Constructor

            T -- pseudo temperature used for updating
        '''
        super(StochasticHopfield, self).__init__(max_size)
        self.psuedo_temp = T

    def _updating_rule(self, h_i):
        ''' Uses specified updating rule

            h_i -- the intermediate hebbian of node id
            Returns -- the value of the node
        '''
        prob = 1. / (1. + np.exp(-2.*h_i / self.psuedo_temp))
        return 1 if np.random.uniform() <= prob else -1
