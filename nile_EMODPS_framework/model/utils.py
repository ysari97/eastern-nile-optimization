# A file generated mostly to modify the interpolation such that we don't need to do iscomplex check all the time!

import numpy as np
from numpy.core.multiarray import (
    interp as compiled_interp
    )


def modified_interp(x, xp, fp, left=None, right=None):
    fp = np.asarray(fp)

    interp_func = compiled_interp
    return interp_func(x, xp, fp, left, right)
