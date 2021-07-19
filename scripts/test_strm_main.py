import numpy as np
from src.streamline import Streamline
from scipy.integrate import solve_ivp
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
import sphere_processing as sphere
from numba import jit, prange


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

            p = np.dot(a, np.cross(b, c))
            an = np.sqrt(np.dot(a, a))
            bn = np.sqrt(np.dot(b, b))
            cn = np.sqrt(np.dot(c, c))
            dn = np.sqrt(np.dot(d, d))

            d1 = an * bn * cn + np.dot(a, b) * cn + np.dot(b, c) * an + np.dot(c, a) * bn
            d2 = an * dn * cn + np.dot(a, d) * cn + np.dot(d, c) * an + np.dot(c, a) * dn

            λ += (np.arctan2(p, d1) + np.arctan2(p, d2))
    return λ / (2 * np.pi)


if __name__ == '__main__':
    # Load velocity field
    with np.load('data.npz') as file:
        ug = file['u']
    shape = ug.shape[1:]
    dtype = ug.dtype
    c, d, b = sphere.build_ball(shape, dtype=dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u['g'] = ug

    # Compute cartesian vorticity components
    curl = operators.Curl
    dot = arithmetic.DotProduct
    ω = curl(u)
    ex, ey, ez = sphere.build_cartesian_unit_vectors(b)
    ωx = dot(ex, ω).evaluate()
    ωy = dot(ey, ω).evaluate()
    ωz = dot(ez, ω).evaluate()
    # Parameters
    interp_scales = 4  # Controls refinement of spectral sampling prior to linear interpolation

    # Build cartesian vorticity interpolator
    ω_interp = sphere.build_cartesian_interpolator(ωx, ωy, ωz, scales=interp_scales)
    N_sample = 5

    u = 2 * np.random.rand(N_sample) - 1
    p = 2 * np.pi * np.random.rand(N_sample)
    r = np.random.rand(N_sample) ** (1 / 3.)

    x_0 = r * np.cos(p) * np.sqrt(1 - u * u)
    y_0 = r * np.sin(p) * np.sqrt(1 - u * u)
    z_0 = r * u

    t_bounds = [0, 10]
    tols = (1e-12, 1e-8)

    points = [np.array((x_0[i], y_0[i], z_0[i])) for i in range(N_sample)]

    def wrap_streams(x):
        s = Streamline(x, t_bounds, tols, ω_interp, normalized=True)
        s.integrate()
        return s

    # uncomment for joblib streams.
    # streamlines = Parallel(n_jobs=num_cores)(delayed(parallel_streams)(x) for x in points)
    streamlines = [wrap_streams(x) for x in points]

    λ = np.zeros((N_sample, N_sample))
    for i in range(1, N_sample):
        s1 = streamlines[i]
        ell = s1.sol.y.T
        for j in range(i):
            s2 = streamlines[j]
            k = s2.sol.y.T
            λ[i, j] = compute_link_DS(ell, k)
            λ[j, i] = λ[i, j]
    # save output
    print(λ)
    with open('test.npy', 'wb') as file:
        np.save(file, λ)
