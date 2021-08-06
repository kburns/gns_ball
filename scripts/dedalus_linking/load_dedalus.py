

import numpy as np
import sphere_processing as sphere
from dedalus.core import field, operators, arithmetic


def load_field(filename, fieldname):
    with np.load(filename) as file:
        data = file[fieldname]
    shape = data.shape[1:]
    dtype = data.dtype
    c, d, b = sphere.build_ball(shape, dtype=dtype)
    f = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    f['g'] = data
    return f


def load_vorticity(filename):
    u = load_field(filename, 'u')
    ω = operators.Curl(u).evaluate()
    return ω


def build_cartesian_vorticity_interpolator(filename, interp_scales=1):
    # Load vorticity field
    ω = load_vorticity(filename)
    # Compute cartesian components
    dot = arithmetic.DotProduct
    ex, ey, ez = sphere.build_cartesian_unit_vectors(ω.domain.bases[0])
    ωx = dot(ex, ω).evaluate()
    ωy = dot(ey, ω).evaluate()
    ωz = dot(ez, ω).evaluate()
    # Return cartesian interpolator
    ω_interp = sphere.build_cartesian_interpolator(ωx, ωy, ωz, scales=interp_scales)
    return ω_interp

