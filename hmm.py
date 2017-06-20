import numpy as np
import copy

class HMM:
    def __init__(self, n_state, n_obs):
        '''
        Args:
            n_state: a number of hidden states.
            n_obs: a number of possible observations.
        '''
        self.n_state = n_state
        self.n_obs = n_obs
        self.A = np.zeros((n_state, n_state))
        self.B = np.zeros((n_state, n_obs))
        self.pi = np.zeros(n_state)

    def init(self, A=None, B=None, pi=None):
        '''
        Args:
            self.A: a matrix of size (n_states, n_states) which represents the
                    state to state transition probabilities.
            self.B: a matrix of size (n_states, n_obs) where n_obs is the number
                    of possible outputs that can be observed, which represents
                    the emission probabilities of respective states for given
                    possible outputs.
            self.pi: a vector of initial probabilities of size n_states that the
                     HMM will start in a given state.
        '''
        if A is not None:
            self.A = A
        else:
            self.A = np.random.uniform(0.0, 1.0, size=self.A.shape)
            self.A /= self.A.sum(axis=1)[:, np.newaxis]
        if B is not None:
            self.B = B
        else:
            self.B = np.random.uniform(0.0, 1.0, size=self.B.shape)
            self.B /= self.B.sum(axis=1)[:, np.newaxis]
        if pi is not None:
            self.pi = pi
        else:
            self.pi = np.random.uniform(0.0, 1.0, size=self.pi.shape)
            self.pi /= self.pi.sum(axis=0)

    def train(self, seqs, n_iter=10, init_random=True):
        '''
        Args:
            seqs: a vector of sequences of observations.
        '''
        if init_random:
            self.init()

        for iter in range(n_iter):
            print('Iteration {}:'.format(iter))

            # Expectation (E) part
            gammas = []
            xis = []
            for s in seqs:
                alphas = self.forward(s)
                betas = self.backward(s)

                gammas.append(self.gammas_from_ab(alphas, betas))
                xis.append(self.xis_from_ab_seq(s, alphas, betas))

            prevA = copy.copy(self.A)
            prevB = copy.copy(self.B)

            # Maximization (M) step
            # Update the A matrix
            for i in range(self.n_state):
                for j in range(self.n_state):
                    s_xis = 0
                    s_gammas = 0
                    for l, seq in enumerate(seqs):
                        for k in range(len(seq) - 1):
                            s_xis += xis[l][k][i][j]
                            s_gammas += gammas[l][k][i]

                    self.A[i, j] = float(s_xis) / denom(float(s_gammas))

            # Update the B matrix
            for i in range(self.n_state):
                s_total = 0
                for l, seq in enumerate(seqs):
                    for k in range(len(seq)):
                        s_total += gammas[l][k][i]

                for n in range(self.n_obs):
                    s_filtered = 0
                    for l, seq in enumerate(seqs):
                        for k in range(len(seq)):
                            if seq[k] == n:
                                s_filtered += gammas[l][k][i]
                    self.B[i, n] = float(s_filtered) / denom(float(s_total))

            # Update the pi vector
            for i in range(self.n_state):
                N = len(seqs)
                self.pi[i] = sum([gammas[n][0][i] for n in range(N)])
            self.pi[:] = self.pi[:] / denom(self.pi[:].sum())

            diffA = np.absolute(np.linalg.norm(self.A, ord='fro') -
                                np.linalg.norm(prevA, ord='fro'))
            diffB = np.absolute(np.linalg.norm(self.B, ord='fro') -
                                np.linalg.norm(prevB, ord='fro'))
            print("Diff A: {}".format(diffA))
            print("Diff B: {}".format(diffB))

    def forward(self, seq):
        '''
        Args:
            seq: a vector representing a sequence of observed outputs.

        Returns:
            alphas: matrix of size (n_states, K) where n_states is the number
            of states of a HMM and K is the length of the observed sequence.
        '''
        alphas = np.array([[self.pi[i] * self.B[i][seq[0]]
            for i in range(len(self.pi))]]).T

        for k in range(len(seq) - 1):
            a = np.array([
                sum([alphas[i][k] * self.A[i][j] for i in range(len(self.A))]) *
                self.B[j][seq[k+1]] for j in range(len(self.A))])
            alphas = np.column_stack((alphas, a))

        return alphas

    def backward(self, seq):
        '''
        Args:
            seq: a vector representing a sequence of observed outputs.

        Returns:
            betas: matrix of size (n_states, K) where n_states is the number
            of states of a HMM and K is the length of the observed sequence.
        '''
        betas = np.array([[1 for i in range(len(self.pi))]]).T

        for k in range(len(seq) - 1, 0, -1):
            b =  np.array([
                sum([betas[i][0] * self.A[j][i] * self.B[i][seq[k]]
                    for i in range(len(self.A))]) for j in range(len(self.A))])
            betas = np.column_stack((b, betas))

        return betas

    def gammas_from_ab(self, alphas, betas):
        '''
        Args:
            alphas: a matrix returned from the forward function
            betas: a matrix returned from the backward function

        Returns:
            gammas: matrix of size (K, n_states) where n_states is the number
            of states of a HMM and K is the length of the observed sequence.
        '''
        gammas = np.array([[
            (alphas[i][t] * betas[i][t]) /
            denom(sum(alphas[j][t] * betas[j][t] for j in range(len(self.A))))
            for i in range(len(self.A))] for t in range(alphas.shape[1])])

        return gammas

    def xis_from_ab_seq(self, s, alphas, betas):
        '''
        Args:
            s: current state
            alphas: a matrix returned from the forward function
            betas: a matrix returned from the backward function

        Returns:
            xis: xis for EM
        '''
        K = len(s)
        xis = np.zeros((K-1, self.n_state, self.n_state))
        for k in range(K-1):
            for i in range(self.n_state):
                for j in range(self.n_state):
                    xis[k][i][j] = (alphas[i, k] * self.A[i, j] *
                                    betas[j, k+1] * self.B[j, s[k+1]])
            det = xis[k].sum(axis=0).sum(axis=0)
            xis[k] /= denom(det)
        return xis

    def viterbi(self, seq):
        '''
        Args:
            seq: a vector representing a sequence of observed outputs.

        Returns:
            probabl_seq: a vector of length seq which contains the most
            probable sequence of states that generated seq.
        '''
        delta = np.zeros((len(self.pi),len(seq)))
        delta_arg = np.zeros((len(self.pi),len(seq)))
        delta[0] = self.pi * self.B.T[seq[0]]

        for k in range(1, len(seq)):
            delta[k] = np.array([
                max(self.A.T[j] * self.B[j][seq[k]] * delta[k-1])
                for j in range(len(self.A))])
            delta_arg[k] = np.array([
                np.argmax(self.A.T[j] * self.B[j][seq[k]] * delta[k-1])
                for j in range(len(self.A))])

        probabl_seq = np.zeros(len(seq))
        probabl_seq[len(seq)-1] = np.argmax(delta[len(seq)-1])

        for i in range(len(seq)-1, 0, -1):
            probabl_seq[i-1] = delta_arg[i][probabl_seq[i]]

        return probabl_seq

def denom(denominator):
    if denominator == 0:
        return 0.0001
    return denominator
