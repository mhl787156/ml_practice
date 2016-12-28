import numpy as np

class HopfieldBasic:
    """Basic Hopfield Network"""

    def __init__(self, size):
        ''' Creates a basic one pattern hopfield network 

            size -- the number of nodes in the network
        '''
        self.size = size
        self._initialise_nodes()
        self._initialise_weights()

    def _initialise_weights(self):
        ''' Initialises the weights to random values in [0, 1]
            Weight matrix is symmetrical with w_ii = 0 forall i
        '''
        w = np.random.rand(self.size, self.size)
        w = (w + w.T) / 2
        np.fill_diagonal(w, 0)
        self.w = w

    def _initialise_nodes(self):
        ''' Initialises the nodes randomly to -1 or 1 '''
        self.n = np.random.choice([-1, 1], self.size)

    def _train_weights(self, v):
        ''' trains all the weights using the input vector '''
        self.w = np.outer(v, v)                 # w_ij = v_i * v_j
        np.fill_diagonal(self.w, 0)             
    
    def _update_node(self, i):
        ''' Updates a node i with the weights, may flip the node values

            i -- ith node
        '''
        # print '     w_i:', self.w[i,]
        # print '     n_i:', self.n
        h_i = np.dot(self.w[i,], self.n)        # h_i = sum(j=1-n) w_ij * n_j
        # print '     h_i:', h_i
        self.n[i] = 1 if h_i > 0 else -1        # n_i = sgn(h_i)

    def _update(self, v, time=50000, interval = 0):
        ''' Updates the hopfield network 

            v -- the initial vector
            time -- the number of iterations to run for
        '''
        self.n = v
        state = []
        for t in range(time):
            i = np.random.randint(0, self.size - 1)
            self._update_node(i)
            if interval != 0 and t % interval == 0:
                state.append(np.array(self.n))
        return state
            


    def calculate_energy(self):
        ''' Calculates and returns the energy of the network.

            returns -- E = -(1/2)sum(i,j) w_ij * n_i * n_j = -(1/2) N^T * W * N
        '''
        return - (1/2) * np.sum(self.n[:,None] * self.w * self.n)

    def train(self, v):
        ''' Trains the network with an input vector '''
        self._train_weights(v)

    def test_with_random(self):
        ''' Tests the network with a random image '''
        test = np.random.choice([-1, 1], self.size)
        log = self._update(test, time=5000, interval=500)
        log.append(self.n)
        return log


    

