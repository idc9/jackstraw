import numpy as np

from jackstraw.utils import pca, svd_wrapper


def permutationPA(X, B=100, alpha=0.05, method='pca', max_rank=None):
    """
    Estimates the number of significant principal components using a permutation test.
    Adapted from https://github.com/ncchung/jackstraw and Buja and Eyuboglu (1992).

    Buja A and Eyuboglu N. (1992) Remarks on parrallel analysis. Multivariate Behavioral Research, 27(4), 509-540

    Parameters
    ----------
    X: data matrix n x d

    B (int): number of permutations

    alpha (float): cutoff value

    method (str): one of ['pca', 'svd']

    max_rank (None, int): will compute partial SVD with this rank to save
    computational time. If None, will compute full SVD.
    """

    if method == 'pca':
        decomp = pca
    elif method == 'svd':
        decomp = svd_wrapper
    else:
        raise ValueError('{} is invalid method'.format(method))

    # compute eigenvalues of observed data
    U, D, V = decomp(X, rank=max_rank)

    # squared frobinius norm of the matrix, also equal to the
    # sum of squared eigenvalues
    frob_sq = (X**2).sum()
    dstat_obs = (D**2)/frob_sq

    # compute premutation eigenvalues
    dstat_null = np.zeros((B, len(D)))
    for b in range(B):
        X_perm = np.apply_along_axis(np.random.permutation, 0, X)
        U_perm, D_perm, V_perm = decomp(X_perm, rank=max_rank)
        dstat_null[b, :] = (D_perm**2)/frob_sq

    # compute p values
    pvals = np.ones(len(dstat_obs))
    for i in range(len(dstat_obs)):
        pvals[i] = np.mean(dstat_null[:, i] >= dstat_obs[i])
    for i in range(1, len(dstat_obs)):
        pvals[i] = max(pvals[i - 1], pvals[i])

    # estimate rank
    r_est = sum(pvals <= alpha)

    return r_est, pvals
