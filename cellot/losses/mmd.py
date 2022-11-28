""" Implementation of the MMD distance.
"""


import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def mmd_distance(x:np.ndarray, y:np.ndarray, gamma:float) -> float:
    """Computes the MMD distance between two distributions.  

    MMD refers to the kernel maximum mean discrepancy, a metric to measure distances between distributions.
    Reports an unbiased estimate where expectations are evaluated by averages over the cells in each set.
    The RBF kernel is employed.

    Args:
        x: a np.ndarray representing samples of a random variable with distribution p.
        y: a np.ndarray representing samples of a random variable with distribution q.
        gamma: a float representing the length scale of the RBF kernel.
    
    Returns:
        A float representing the MMD distance between p and q.
    """

    # Computes the estimates of the expectation of phi(x,x').
    xx = rbf_kernel(x, x, gamma)
    # Computes the estimates of the expectation of phi(x,y).
    xy = rbf_kernel(x, y, gamma)
    # Computes the estimates of the expectation of phi(y,y').
    yy = rbf_kernel(y, y, gamma)

    # Computes and returns MMD distance.
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(target:np.ndarray, transport:np.ndarray, gammas=None) -> float:
    """Computes the MMD distance between two distributions.  

    MMD refers to the kernel maximum mean discrepancy, a metric to measure distances between distributions.
    Reports an unbiased estimate where expectations are evaluated by averages over the cells in each set.
    The RBF kernel is employed.

    Args:
        target: a np.ndarray representing samples drawn from the target distribution p.
        transport: a np.ndarray representing samples drawn from the predicted transport distribution q.
        gammas: a list of float representing the length scales used for the RBF kernel.
    
    Returns:
        A float representing the MMD distance between p and q averaged on different gammas.
    """

    # If no gammas input, uses default list.
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    # An enhanced version of function mmd_distance() to avoid numerical errors.
    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    # For every length scale, computes the mmd distance and then returns the average.
    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))
