jackstraw
----

**author**: `Iain Carmichael`_

Additional documentation, examples and code revisions are coming soon.
For questions, issues or feature requests please reach out to Iain:
iain@unc.edu.

Overview
========

This package performs association tests between the observed data and their systematic patterns of variation. This package implements methods from (Chung and Storey, 2015) in python. For an R version (which we followed closely) see https://github.com/ncchung/jackstraw.

*Chung, N.C.* and *Storey, J.D.* (2015) Statistical significance of variables driving systematic variation in high-dimensional data. Bioinformatics, 31(4): 545-554
http://bioinformatics.oxfordjournals.org/content/31/4/545

Installation
============
This is currently an informal package under development so I've only made it installable from github.

::

    git clone https://github.com/idc9/jackstraw.git
    python setup.py install

Example
=======

.. code:: python
    import numpy as np
    from jackstraw.jackstraw import Jackstraw

    X = np.random.normal(size=(100, 20))
    jack = Jackstraw(S = 10, B = 100)
    jack.fit(X, method='pca', rank=4)
    jack.rejected

    array([], dtype=int64)

For some more example code see `these example notebooks`_.

Help and Support
================

Additional documentation, examples and code revisions are coming soon.
For questions, issues or feature requests please reach out to Iain:
iain@unc.edu.

Documentation
^^^^^^^^^^^^^

The source code is located on github: https://github.com/idc9/jackstraw.

Testing
^^^^^^^

Testing is done using `nose`.

Contributing
^^^^^^^^^^^^

We welcome contributions to make this a stronger package: data examples,
bug fixes, spelling errors, new features, etc.



.. _Iain Carmichael: https://idc9.github.io/
.. _these example notebooks: https://github.com/idc9/jackstraw/tree/master/doc
