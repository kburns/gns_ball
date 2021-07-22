

import numpy as np
from numba import jit, prange


def compute_link_DS(s1, s2):
    """Compute linking number between two Streamline objects."""
    # Make closed polylines from streamline nodes
    x1 = np.vstack((s1.x, s1.x[0:1]))
    x2 = np.vstack((s2.x, s2.x[0:1]))
    # Call compiled linking number computation
    return _compute_link_DS(x1, x2)


def compute_partial_link_DS(s1, s2, dL=1):
    """Compute partial linking numbers between two Streamline objects."""
    pass


@jit(nopython=True, parallel=True)
def _compute_link_DS(ls, ks):
    """
    Compute linking number between two closed polylines with shape (N, 3).
    The last point should equal the first point for each line.
    """
    λ = 0
    Nl = ls.shape[0]
    Nk = ks.shape[0]
    for i in prange(Nk-1):
        for j in range(Nl-1):
            a = ls[j, :] - ks[i, :]
            b = ls[j, :] - ks[i + 1, :]
            c = ls[j + 1, :] - ks[i + 1, :]
            d = ls[j + 1, :] - ks[i, :]
            p = np.dot(a, np.cross(b, c))
            an = np.sqrt(np.dot(a, a))
            bn = np.sqrt(np.dot(b, b))
            cn = np.sqrt(np.dot(c, c))
            dn = np.sqrt(np.dot(d, d))
            d1 = an * bn * cn + np.dot(a, b) * cn + np.dot(b, c) * an + np.dot(c, a) * bn
            d2 = an * dn * cn + np.dot(a, d) * cn + np.dot(d, c) * an + np.dot(c, a) * dn
            λ += (np.arctan2(p, d1) + np.arctan2(p, d2))
    return λ / (2 * np.pi)

