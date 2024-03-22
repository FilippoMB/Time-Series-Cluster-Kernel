import numpy as np
from scipy.stats import norm

def GMMposterior(X, C, mu, s2, theta, dim_idx, time_idx, missing):
    """
    Evaluate the posterior for the data X of the GMM described by C, mu, s2, and theta.
    
    Parameters:
    X: Data array of size N x V x T
    C: Number of mixture components
    mu: Cluster means over time and variables (V x T)
    s2: Cluster stds over variables (V x 1)
    theta: Cluster priors
    dim_idx: Subset of variables to be used in the clustering
    time_idx: Subset of time intervals to be used in the clustering
    missing: Binary indicator. 1 if there is missing data and 0 if not
    
    Returns:
    Q: Posterior probabilities
    """
    
    N = X.shape[0]  # Number of time series
    Q = np.zeros((N, C))
    sX = X[:, time_idx, :][:, :, dim_idx]
    sV = len(dim_idx)
    sT = len(time_idx)

    if missing == 1:
        nan_idx = np.isnan(sX)
        R = np.ones(sX.shape)
        R[nan_idx] = 0
        sX[nan_idx] = -100000

        for c in range(C):
            distr_c = norm.pdf(sX, np.transpose(np.tile(mu[:, :, c][...,None], (1, 1, N)), (2, 0, 1)),
                               np.transpose(np.tile(np.sqrt(s2[:, c])[...,None,None], (1, N, sT)), (1, 2, 0))) ** R
            low_prob_threshold = norm.pdf(3)
            distr_c[distr_c < low_prob_threshold] = low_prob_threshold
            distr_c = distr_c.reshape(N, sV*sT)
            Q[:, c] = theta[c] * np.prod(distr_c, axis=1)
        Q = Q / np.sum(Q, axis=1, keepdims=True)

    elif missing == 0:
        for c in range(C):
            distr_c = norm.pdf(sX, np.transpose(np.tile(mu[:, :, c][...,None], (1, 1, N)), (2, 0, 1)),
                               np.transpose(np.tile(np.sqrt(s2[:, c])[...,None,None], (1, N, sT)), (1, 2, 0)))
            low_prob_threshold = norm.pdf(3)
            distr_c[distr_c < low_prob_threshold] = low_prob_threshold
            distr_c = distr_c.reshape(N, sV*sT)
            Q[:, c] = theta[c] * np.prod(distr_c, axis=1)
        Q = Q / np.sum(Q, axis=1, keepdims=True)

    else:
        raise ValueError('The value of the variable missing is not 0 or 1')

    return Q