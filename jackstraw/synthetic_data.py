import numpy as np


def sample_rjack_ex31(std=1):
    """
    Samples the synthetic data set from example 3.1 from
    https://cran.r-project.org/web/packages/jackstraw/vignettes/jackstraw.pdf
    Note observations are on rows here!

    We simulate a simple gene expression data with m = 1000 genes (or variables)
    and n = 20 samples (or observations). The following simulation code
    generates a dichonomous mean shift between the first set of 10 samples and the
    second set of 10 samples (e.g., this may represent a case-control study). This
    mean shift affects 10% of m = 1000 genes:

    (most of) the first 100 variables should truely play a role in the first PC while
    none of the remaining variables should.

    Parameters
    ----------
    std (float, int): noise standard deviation

    Returns
    -------
    Y, LB

    Y: the observed data matrix with 20 observations (rows) and
    1000 variabiles (columns)

    LB: the signal matrix without noise
    """
    B = np.concatenate([np.random.uniform(size=100, low=.1, high=1),
                        np.zeros(900)])
    L = np.concatenate([[1]*10, [-1]*10])
    L = L/np.std(L)
    E = np.random.normal(size=(20, 1000), scale=std)
    LB = np.outer(L, B)
    Y = LB + E
    return Y, LB
