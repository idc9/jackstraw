from statsmodels.stats.multitest import multipletests
import numpy as np
from sklearn.externals.joblib import dump, load
import statsmodels.api as sm
from math import ceil
from jackstraw.utils import pca, svd_wrapper


class Jackstraw(object):
    def __init__(self, S=None, B=None, multitest_method='fdr_bh', alpha=0.05,
                 seed=None):
        """
        Computes p values for tests of linear association between variables and
        principal components (and other latent variable linear dimensionality reduction
        procedures). For a data matrix X in R^(n x d) (n observations on the rows,
        d variables on the columns), it's rank k PCA is provided by

        U, D, V = rank-k-SVD(X_centered)

        where U in R^(n x k) is the matrix of scores (left singluar values or
        latent variables) and V in R^(d x k) is the matrix of loadings. The goal
        is to test H0: V[i, :] = 0 or possibly H0: V[i, comps] = 0 (where comps
        is some subset of [k]).

        For additional details see:

        https://cran.r-project.org/web/packages/jackstraw/index.html

        Chung, N.C. and Storey, J.D.(2015) Statistical significance of variables driving systematic variation in high-dimensional data. Bioinformatics, 31(4): 545-554
http://bioinformatics.oxfordjournals.org/content/31/4/545

        Parameters
        ----------
        S (int): number of synthetic null variables to sample. If None will be
        set by defualt to d/10.

        B (int): number of times to resample. If None will be set by default
        to d * 10 / s.

        multitest_method (str, None): method to use for correcting for multiple
        testing using statsmodels.stats.multitest.multipletests. If None, no
        correction is used, otherwise should be one of
        ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg',
        'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']. See statsmodels
        for details.

        alpha (float): cutoff for rejection.

        seed (int): seed for random number generator.


        Attributes
        ----------
        F_obs: observed F-statistics for each of the d variables

        F_null: sampled F-statistics from the null distribution.

        pvals_raw: raw p-values.

        pvals_adj: p-values after adjusting for multiple testing.

        rejected: list of rejected variables indices.
        """
        self.S = S
        self.B = B
        self.multitest_method = multitest_method
        self.alpha = alpha
        self.seed = seed

        self.F_obs = None
        self.F_null = None
        self.pvals_raw = None
        self.pvals_adj = None
        self.rejected = None

    def save(self, fpath, compress=9):
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        return load(fpath)

    def get_scores(self, X, method, rank):

        if callable(method):
            return method(X, rank=rank)

        elif method == 'pca':
            U, D, V = pca(X, rank=rank)
            return U

        elif method == 'svd':
            U, D, V = svd_wrapper(X, rank=rank)
            return U

        else:
            raise ValueError('{} is not a valid argument for method'.format(method))

    def fit(self, X, method, rank, comps='all'):
        """

        Parameters
        ----------
        X: data matrx with n observations (rows) and d variables (columns)

        method (str, callable): how to compute scores. If ['pca', 'svd'] will use
        the unnormalized scores (i.e. the left singular vectors). The user can
        provide an arbitrary function to compute scores. This function should
        accept two arguments (X, rank) and should return a single n x rank
        numpy array of scores.

        rank (int): rank of the decomposition to compute.

        comps (None, list of ints): a subset of components of interest.If 'all'
        will include all components.
        """
        X = np.array(X)
        n, d = X.shape
        assert rank <= d

        if self.S is None:
            self.S = ceil(d/10)
        if self.B is None:
            self.B = ceil(d * 10/self.S)

        if comps == 'all':
            comps = list(range(rank))
        elif type(comps) in [float, int]:
            comps = [int(comps)]

        if self.seed:
            np.random.seed(self.seed)

        # compute observed F stats for each variable
        scores = self.get_scores(X, method, rank)
        self.F_obs = np.zeros(d)
        for j in range(d):
            self.F_obs[j] = get_F_stat(response=X[:, j],
                                       explanatory=scores,
                                       in_H0=comps)

        # compute null F-stats
        self.F_null = np.zeros((self.S, self.B))
        for b in range(self.B):
            X_perm = X.copy()

            # permute observations of S randomly selected variables
            vars_to_perm = np.random.choice(d, size=self.S, replace=False)
            for j in vars_to_perm:
                X_perm[:, j] = np.random.permutation(X[:, j])

            # compute PCA of permuted matrix
            scores_perm = self.get_scores(X_perm, method, rank)

            # compute F stats
            for s, var in enumerate(vars_to_perm):
                self.F_null[s, b] = get_F_stat(response=X_perm[:, var],
                                               explanatory=scores_perm,
                                               in_H0=comps)

        self.pvals_raw, self.pvals_adj, self.rejected = \
            jackstraw_hypothesis_tests(self.F_obs, self.F_null,
                                       method=self.multitest_method,
                                       alpha=self.alpha)


def get_F_stat(response, explanatory, in_H0=None):
    """
    run linear regression model for
    y = x^t beta + eps
    where x is a d vector and y is a scalar.

    response: vector with n observations
    explanatory: n x d matrix

    in_H0 (None, list): subset of variables to test equal to zero
    i.e. H0: beta[i] = 0, for i in in_H0. If in_H0 is None then
    tests if all variables are zero i.e. H0: beta = 0

    Returns the F statistic of H0
    """

    if in_H0 is None:
        in_H0 = list(range(explanatory.shape[1]))

    model = sm.OLS(response, sm.add_constant(explanatory)).fit()

    # build restriction matrix for H0: R beta = 0
    r_matrix = np.identity(len(model.params))[1:, :]  # include every variable
    to_remove = list(set(range(explanatory.shape[1])).difference(in_H0))
    if len(to_remove) > 0:
        r_matrix = np.delete(r_matrix, to_remove, axis=0)
    return model.f_test(r_matrix).fvalue.item()


def jackstraw_hypothesis_tests(F_obs, F_null, method='fdr_bh', alpha=0.05):
    """

    Parameters
    ----------
    F_obs (list): the observed F-statistics for each varialbe

    F_null (array): the sampled, null F-statistcs for each variable.

    method (None, str): how to correct for multiple testing. See statsmodels.stats.multitest.multipletests for method options.

    alpha (float): cutoff for rejection.
    """

    # compute p-vals
    pvals_raw = np.zeros(len(F_obs))
    for j in range(len(F_obs)):
        pvals_raw[j] = np.mean(F_obs[j] <= F_null)

    # adjust for multiple testing
    if method is None:  # no adjustment
        pvals_adj = pvals_raw
        rejected = pvals_raw <= alpha
    else:
        rejected, pvals_adj, _, __ = multipletests(pvals_raw, method=method, alpha=alpha)

    rejected = np.where(rejected)[0]

    return pvals_raw, pvals_adj, rejected
