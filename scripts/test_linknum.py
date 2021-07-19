import numpy as np
from numba import jit, prange
import time


@jit(nopython=True, parallel=True)
def compute_link_DS(l, k):
    """ Computes the linking number between 2 polylines - uses JIT compiling
    via numba and auto-parallelization to get faster """
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

            cr = np.cross(b, c)
            p = np.dot(a, cr)
            an = np.sqrt(np.dot(a, a))
            bn = np.sqrt(np.dot(b, b))
            cn = np.sqrt(np.dot(c, c))
            dn = np.sqrt(np.dot(d, d))

            d1 = an * bn * cn + np.dot(a, b) * cn + np.dot(b, c) * an + np.dot(c, a) * bn
            d2 = an * dn * cn + np.dot(a, d) * cn + np.dot(d, c) * an + np.dot(c, a) * dn

            λ += (np.arctan2(p, d1) + np.arctan2(p, d2))
    return λ / (2 * np.pi)


if __name__ == '__main__':
    testdata = {}
    N = 243
    eps = 0  # noise
    # discretize two circles
    s = np.pi * np.linspace(-1, 1, N)
    t = s
    # define the embeddings in R^3
    # exact formulas for curves with linking number n
    n = 10
    amp = 2
    fx = 3 * np.cos(s[:-1:2])
    fy = 3 * np.sin(s[:-1:2])
    fz = 0 * s[:-1:2]

    gx = (3 + amp * np.sin(n * s)) * np.cos(t) + eps * amp * np.random.randn(*s.shape)
    gy = (3 + amp * np.sin(n * t)) * np.sin(t) + eps * amp * np.random.randn(*s.shape)
    gz = amp * np.cos(n * t) + eps * amp * np.random.randn(*s.shape)
    testdata['wrapped'] = [np.vstack((fx, fy, fz)).T, np.vstack((gx, gy, gz)).T]
    Lwrapped = compute_link_DS(*testdata['wrapped'])
    tic = time.perf_counter()
    for _ in range(10):
        Lwrapped = compute_link_DS(*testdata['wrapped'])
    toc = time.perf_counter()
    print(f"Linking number = {Lwrapped:0.9f} in {(toc - tic)/10:0.4f} seconds")
