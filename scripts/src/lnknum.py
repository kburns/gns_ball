import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def compute_link_DS(l, k):
    """ Computes the linking number between 2 polylines - uses JIT compiling
    via numba and auto-parallelization to get faster 
    l,k must have shape (Nl, 3) and (Nk, 3)
    """
    λ = 0
    Nl = l.shape[0]
    Nk = k.shape[0]
    # Add first point to close loop
    ls0 = np.expand_dims(l[0, :], 0)
    ks0 = np.expand_dims(k[0, :], 0)
    ls = np.vstack((l, ls0))
    ks = np.vstack((k, ks0))
    for i in prange(Nk):
        for j in range(Nl):
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
