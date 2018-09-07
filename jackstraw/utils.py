from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd
from sklearn.preprocessing import scale
import numpy as np


def svd_wrapper(X, rank=None):
    """
    Computes the (possibly partial) SVD of a matrix.

    Parameters
    ----------
    X: either dense or sparse
    rank: rank of the desired SVD (required for sparse matrices)

    Output
    ------
    U, D, V
    the columns of U are the left singular vectors
    the COLUMNS of V are the left singular vectors

    """

    if rank is not None and rank > min(X.shape):
        raise ValueError('rank must be <= the smallest dimension of X. rank= {} was passed in while X.shape = {}'.format(rank, X.shape))

    if rank is None or rank == min(X.shape):
        U, D, V = full_svd(X, full_matrices=False)
        V = V.T
    else:
        scipy_svds = svds(X, rank)
        U, D, V = fix_scipy_svds(scipy_svds)

    return U, D, V


def fix_scipy_svds(scipy_svds):
    """
    scipy.sparse.linalg.svds orders the singular values backwards,
    this function fixes this insanity and returns the singular values
    in decreasing order

    Parameters
    ----------
    scipy_svds: the out put from scipy.sparse.linalg.svds

    Output
    ------
    U, D, V
    ordered in decreasing singular values
    """
    U, D, V = scipy_svds

    sv_reordering = np.argsort(-D)

    U = U[:, sv_reordering]
    D = D[sv_reordering]
    V = V.T[:, sv_reordering]

    return U, D, V


def mean_center(X):
    return scale(X, with_mean=True, with_std=False)


def pca(X, rank=None):
    """
    Computes the PCA of X where the observations are on the rows. Suppose X is (n x d) (n observations) and r = min(m, d, rank) then

    U (n x r): the scores
    D (r x r): the singular values
    V (d x r): the loadings

    Parameters
    ----------
    X (numpy matrix/array): the data matrix

    rank (None, int): the number of PCs to compute. If None, will compute
    the full PCA

    Output
    ------
    U, D, V

    """
    # m = np.asarray(X.mean(axis=0)).reshape(-1)
    # m = X.mean(axis=0)
    # X_cent = X - np.outer(np.ones((X.shape[0],)), m)
    return svd_wrapper(mean_center(X), rank)
