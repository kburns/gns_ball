import numpy as np
from src.streamline import Streamline
from src.lnknum import compute_link_DS
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
import sphere_processing2 as sphere
import pickle


if __name__ == '__main__':
    ###### Build vorticity field interpolator
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
    # Interpolation parameters
    interp_scales = 4  # Controls refinement of spectral sampling prior to linear interpolation

    # Build cartesian vorticity interpolator
    ω_interp = sphere.build_cartesian_interpolator(ωx, ωy, ωz, scales=interp_scales)
    
    
    # set seed
    np.random.seed(1986) # cf Arnold 1986
    
    ###### Integrates streamlines for various streamline integration tolerances & settings
    # Test parameters
    
    N_sample = 100
    L_bounds = 100
    
    atol = 1e-6
    maxdt = 1e-2
    
    ## Sample initial conditions
    u = 2 * np.random.rand(N_sample) - 1
    p = 2 * np.pi * np.random.rand(N_sample)
    r = np.random.rand(N_sample) ** (1 / 3.)

    x_0 = r * np.cos(p) * np.sqrt(1 - u * u)
    y_0 = r * np.sin(p) * np.sqrt(1 - u * u)
    z_0 = r * u
    
    points = [np.array((x_0[i], y_0[i], z_0[i])) for i in range(N_sample)]
    rtol = 1e-12
    ## integrate streamlines
    def wrap_streams(x0):
        s = Streamline(x0, L_bounds, rtol, atol, maxdt)
        s.integrate(ω_interp)
        return s
    streamlines = [wrap_streams(x) for x in points]
    args = (N_sample, L_bounds, int(-np.log10(maxdt)), int(-np.log10(atol)))
    with open('longT/strms2_N{0}_L{1}_mdt{2}_atol{3}.pickle'.format(*args), 'wb') as file:
        pickle.dump(streamlines, file)
        print("Integration & pickling successful: atol={0}, max_step={1}".format(atol, maxdt))