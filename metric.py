import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def compute_mmd(X, Y):

    m = len(X)
    n = len(Y)
    gamma = 10

    kernel_xx = rbf_kernel(X, X, gamma)
    np.fill_diagonal(kernel_xx, 0)
    sum_xx = np.sum(kernel_xx)

    kernel_yy = rbf_kernel(Y, Y, gamma)
    np.fill_diagonal(kernel_yy, 0)
    sum_yy = np.sum(kernel_yy)

    kernel_xy = rbf_kernel(X, Y, gamma)
    sum_xy = np.sum(kernel_xy)
    mmd2 = (sum_xx / (m * (m - 1))) + (sum_yy / (n * (n - 1))) - (2 * sum_xy / (m * n))
    
    return mmd2