"""Bradley-Terry ranking adapted from cdebacco/ds_prob_inf_net.

This file includes local modifications to the adapted source.
Source: https://github.com/cdebacco/ds_prob_inf_net
License: GNU GPL v3.0; see the repository root LICENSE file.
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm

class BradleyTerry:
    """
    A class implementation of the Bradley-Terry algorithm for computing hierarchical rankings
    from directed networks.
    
    Parameters
    ----------
    alpha : float, default=0
        Regularization parameter. If 0, uses Lagrange multiplier approach.
        If > 0, performs L2 regularization.
        
    Attributes
    ----------
    ranks_ : array-like of shape (n_nodes,)
        The computed ranks for each node after fitting
    is_fitted_ranks_ : bool
        Whether the model has been fitted
    """
    
    def __init__(self, alpha=0):
        self.ranks = None
        self.is_fitted_ranks_ = False
        self.A = None

    def fit(self, A, method='optimize'):
        """
        Compute the BT solution for the input adjacency matrix.
        
        Parameters
        ----------
        A : array-like or sparse matrix of shape (n_nodes, n_nodes)
            The adjacency matrix of the directed network
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validation of A
        if not (A.shape[0] == A.shape[1]):
            raise ValueError("Adjacency matrix must be square")

        neg_entries = A < 0
        if neg_entries.any():
            raise ValueError("Adjacency matrix cannot contain negative entries")

        self.A = A
        self.N = A.shape[0]
        if method == 'optimize':
            self.ranks = self._solve_bt()
        elif method == 'em':
            self.ranks = self.get_scores_bt_em_fast(self.A)

        self.is_fitted_ranks_ = True

        return self

    def bt_likelihood(self,scores):
        log_likelihood = 0
        for i,j in zip(*self.A.nonzero()):

            prob = np.exp(scores[i]) / (np.exp(scores[j]) + np.exp(scores[i]))
            log_likelihood += np.log(prob)
        return -log_likelihood

    def _solve_bt(self):
        init_params = np.zeros(self.N)
        result = minimize(self.bt_likelihood,init_params)
        scores = np.array([result.x[i] for i in np.arange(self.N)])
        return scores


    def predict(self, ij_pair):
        """Predict probability that i -> j in a pair [i,j]"""
        if not self.is_fitted_ranks_:
            raise ValueError("Call fit before predicting")
        i = ij_pair[0]
        j = ij_pair[1]
        if not (0 <= i < self.A.shape[0] and 0 <= j < self.A.shape[0]):
            raise ValueError(f"Indices {i}, {j} out of bounds for matrix of size {self.A.shape[0]}")
        return np.exp(self.scores[w]) / (np.exp(self.scores[i]) + np.exp(self.scores[j]))

    def get_scores_bt_em(self, eps: float = 1e-6):
        '''
        Use EM algorithm
        '''
        # Regularize A to avoid divergences: add 1 win and 1 loss for each player of value eps
        A_reg = self.A + eps
        np.diagonal(A_reg,0)
        df = pd.DataFrame(A_reg, columns=np.arange(self.N), index=np.arange(self.N))
        def get_estimate(i, p, df):
            get_prob = lambda i, j: np.nan if i == j else p.iloc[i] + p.iloc[j]
            n = df.iloc[i].sum()

            d_n = df.iloc[i] + df.iloc[:, i]
            d_d = pd.Series([get_prob(i, j) for j in range(len(p))], index=p.index)
            d = (d_n / d_d).sum()

            return n / d

        def estimate_p(p, df):
            return pd.Series([get_estimate(i, p, df) for i in range(df.shape[0])], index=p.index)

        def iterate(df, p=None, n=20, sorted=False):
            if p is None:
                p = pd.Series([1 for _ in range(df.shape[0])], index=list(df.columns))

            estimates = [p]

            for _ in range(n):
                p = estimate_p(p, df)
                p = p / p.sum()
                estimates.append(p)

            p = p.sort_values(ascending=False) if sorted else p
            return p, pd.DataFrame(estimates)

        p, estimates = iterate(df, n=100)
        return p#, estimates



    def get_scores_bt_em_fast(self, A, eps=1e-6, n_iter=100):
        N = A.shape[0]
        A_reg = A + eps
        np.fill_diagonal(A_reg, 0)
        p = np.ones(N)
        n = A_reg.sum(axis=1)
        for _ in tqdm(range(n_iter), desc="EM fit"):
            denom = p[:, None] + p[None, :]
            np.fill_diagonal(denom, 1)
            d_matrix = (A_reg + A_reg.T) / denom
            np.fill_diagonal(d_matrix, 0)
            d = d_matrix.sum(axis=1)
            p = n / d
            p = p / p.sum()
        return p

    def get_rescaled_ranks(self, target_scale):
        """Rescale ranks using target scale"""
        scaling_factor = 1 / (np.log(target_scale / (1 - target_scale)))
        return self.ranks * scaling_factor