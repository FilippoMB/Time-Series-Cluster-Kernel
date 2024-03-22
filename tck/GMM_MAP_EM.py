import numpy as np
from scipy.stats import norm
from numpy.random import rand, randint
from .GMMposterior import GMMposterior


def GMM_MAP_EM(X, **kwargs):
    """
    Fit a GMM to time series data with missing values using MAP-EM.
    
    Parameters:
    X (numpy.ndarray): Data array of size N x T x V.
    kwargs: Various optional parameters.
    
    Returns:
    tuple: Q, mu, s2, theta, dim_idx, time_idx
    """
    N, T, V = X.shape

    # Handling optional parameters with default values
    minN = kwargs.get('minN', 0.8)
    minV = kwargs.get('minV', 1 if V == 1 else 2)
    maxV = kwargs.get('maxV', V)
    minT = kwargs.get('minT', 6)
    maxT = kwargs.get('maxT', min(int(0.8*T), 25))
    C = kwargs.get('C', 40)
    I = kwargs.get('I', 20)
    missing = kwargs.get('missing', 2)
    
    # Hyperparameters for mean and std dev prior of the mixture components
    a0 = (1.0 - 0.001) * rand() + 0.001
    b0 = (0.2 - 0.005) * rand() + 0.005
    n0 = (0.2 - 0.001) * rand() + 0.001
    
    # Randomly subsample dimensions, time intervals, and samples
    if N > 100:
        sN = randint(round(minN*N), N+1)
    else:
        sN = round(0.9*N)

    sub_idx = np.sort(np.random.choice(N, sN, replace=False))

    sV = randint(minV, maxV+1)
    dim_idx = np.sort(np.random.choice(V, sV, replace=False))

    t1 = randint(0, T - minT+1)
    t2 = randint(t1 + minT, min(T, t1 + maxT)+1)
    sT = t2 - t1
    time_idx = np.arange(t1, t2)  
    sX = X[sub_idx][:, time_idx][:, :, dim_idx]

    # Case with missing data
    if missing == True:

        # Handling missing values
        nan_idx = np.isnan(sX)
        R = np.ones_like(sX)
        R[nan_idx] = 0

        # Calculate empirical moments
        mu_0 = np.nanmean(sX, axis=(0)) # Prior mean over time
        s_0 = np.nanstd(sX, axis=(0, 1), ddof=1)  # Prior std over time and variables
        s2_0 = s_0**2
        
        # Generate covariance matrices and their inverses
        S_0 = np.zeros((sT, sT, sV))
        invS_0 = np.zeros_like(S_0)
        T1, T2 = np.meshgrid(np.arange(1, sT+1), np.arange(1, sT+1), indexing='ij')
        for v in range(sV):
            S_0[:, :, v] = s_0[v] * b0 * np.exp(-a0 * (T1 - T2)**2)
            if np.linalg.cond(S_0[:, :, v]) < 1e-8:
                S_0[:, :, v] += 0.1 * S_0[0, 0, v] * np.eye(sT)
            invS_0[:, :, v] = np.linalg.inv(S_0[:, :, v])

        # Initialize model parameters
        theta = np.ones(C) / C
        mu = np.zeros((sT, sV, C))
        s2 = np.zeros((sV, C))
        Q = np.zeros((sN, C))

        sX[R == 0] = -100000

        for i in range(1, I+1):
            if i == 1:
                # Initialization: random cluster assignment
                cluster = np.random.choice(C, size=sN)
                Q = np.eye(C)[cluster]
            else:
                # Update cluster assignments
                for c in range(C):
                    distr_c = norm.pdf(sX, np.tile(mu[:, :, c], (sN, 1, 1)), np.tile(np.sqrt(s2[:, c])[None, None, :], (sN, sT, 1))) ** R
                    distr_c[distr_c < norm.pdf(3)] = norm.pdf(3)
                    distr_c = distr_c.reshape(sN, sV*sT, order='F')
                    Q[:, c] = theta[c] * np.prod(distr_c, axis=1)
                Q /= Q.sum(axis=1, keepdims=True)

            # Update mu, s2, and theta
            for c in range(C):
                theta[c] = Q[:, c].sum() / sN
                for v in range(sV):
                    var2 = np.sum(R[:, :, v], axis=1).T @ Q[:, c]
                    temp = (sX[:, :, v] - mu[:, v, c][np.newaxis, :])**2
                    var1 = Q[:, c].T @ np.sum(R[:, :, v] * temp, axis=1)
                    s2[v, c] = (n0 * s2_0[v] + var1) / (n0 + var2)
                    A = invS_0[:, :, v] + np.diag(R[:, :, v].T @ Q[:, c] / s2[v, c])
                    b = invS_0[:, :, v] @ mu_0[:, v] + (R[:, :, v] * sX[:, :, v]).T @ Q[:, c] / s2[v, c]
                    mu[:, v, c] = np.linalg.solve(A, b)

        Q = GMMposterior(X, C, mu, s2, theta, dim_idx, time_idx, missing)

    # Case with no missing data
    else:
        # Calculate empirical moments
        mu_0 = np.mean(sX, axis=(0)) # Prior mean over time
        s_0 = np.std(sX, axis=(0, 1), ddof=1)  # Prior std over time and variables
        s2_0 = s_0**2

        # Prepare matrices for spatial correlation
        S_0 = np.zeros((sT, sT, sV))
        invS_0 = np.zeros((sT, sT, sV))
        T1, T2 = np.meshgrid(range(sT), range(sT), indexing='ij')
        for v in range(sV):
            S_0[:, :, v] = s_0[v] * b0 * np.exp(-a0 * (T1 - T2)**2)
            if np.linalg.cond(S_0[:, :, v]) < 1e-8:  # Check if S_0 is invertible
                S_0[:, :, v] += 0.1 * S_0[0, 0, v] * np.eye(sT)
            invS_0[:, :, v] = np.linalg.inv(S_0[:, :, v])   

        # Initialize model parameters
        theta = np.ones(C) / C
        mu = np.zeros((sT, sV, C))
        s2 = np.zeros((sV, C))
        Q = np.zeros((sN, C))

        for i in range(1, I+1):
            if i == 1:
                # Random cluster assignment
                cluster = np.random.randint(C, size=sN)
                Q = np.eye(C)[cluster]

            else:
                for c in range(C):
                    norm_factor = np.sqrt(s2[:, c][np.newaxis, np.newaxis, :])
                    mu_rep = np.repeat(mu[:, :, c][:, :, np.newaxis], sN, axis=2).transpose(2, 0, 1)
                    distr_c = norm.pdf(sX, loc=mu_rep, scale=norm_factor)
                    distr_c[distr_c < norm.pdf(3)] = norm.pdf(3)
                    distr_c = distr_c.reshape(sN, sV*sT)
                    Q[:, c] = theta[c] * np.prod(distr_c, axis=1)
                Q = Q / np.sum(Q, axis=1, keepdims=True)

            # Update mu, s2, and theta
            for c in range(C):
                sumQ = np.sum(Q[:, c])
                theta[c] = sumQ / sN
                for v in range(sV):
                    var2 = sT * sumQ
                    var1 = np.dot(Q[:, c], np.sum((sX[:, :, v] - mu[:, v, c][:, np.newaxis].T) ** 2, axis=1))
                    s2[v, c] = (n0 * s2_0[v] + var1) / (n0 + var2)
                    A = invS_0[:, :, v] + (sumQ / s2[v, c]) * np.eye(sT)
                    b = np.dot(invS_0[:, :, v], mu_0[:, v]) + np.dot(sX[:, :, v].T, Q[:, c]) / s2[v, c]
                    mu[:, v, c] = np.linalg.solve(A, b)

        Q = GMMposterior(X, C, mu, s2, theta, dim_idx, time_idx, missing)

    return Q, mu, s2, theta, dim_idx, time_idx
