# -*- coding: utf-8 -*-


import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.cache import CachedFunction
from dedalus.core.basis import BallBasis
from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI


@CachedFunction
def build_ball(shape, dtype, radius=1, dealias=3 / 2):
    """Build ball basis."""
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, shape, radius=radius, dealias=(dealias, dealias, dealias), dtype=dtype)
    return c, d, b


@CachedFunction
def build_cartesian_unit_vectors(basis):
    """Build cartesian unit vector fields for a spherical basis."""
    b = basis
    d = basis.dist
    c = basis.coordsystem
    dtype = basis.dtype
    phi, theta, r = b.local_grids()
    ex = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    ex['g'][0] = - np.sin(phi)
    ex['g'][1] = np.cos(theta) * np.cos(phi)
    ex['g'][2] = np.sin(theta) * np.cos(phi)
    ey = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    ey['g'][0] = np.cos(phi)
    ey['g'][1] = np.cos(theta) * np.sin(phi)
    ey['g'][2] = np.sin(theta) * np.sin(phi)
    ez = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    ez['g'][0] = 0
    ez['g'][1] = - np.sin(theta)
    ez['g'][2] = np.cos(theta)
    return ex, ey, ez


def extend_spherical_coords(basis, scales=1, reverse_theta=True):
    """Extend spherical coords to include all boundaries."""
    # Fix scales
    if not isinstance(scales, (list, tuple, np.ndarray)):
        scales = (scales,) * 3
    # Build local grids
    phi, theta, r = basis.local_grids(scales)
    # Repeat first phi point
    phi_ext = np.concatenate([phi, [[[2 * np.pi]]]], axis=0)
    # Add theta endpoints
    theta_ext = np.concatenate([[[[np.pi]]], theta, [[[0]]]], axis=1)
    # Add radial endpoints
    if isinstance(basis, BallBasis):
        r_ext = np.concatenate([[[[0]]], r, [[[1]]]], axis=2)
    else:
        raise NotImplementedError("Shell not yet implemented.")
    # Reverse theta
    if reverse_theta:
        theta_ext = theta_ext[:, ::-1, :]
    return phi_ext, theta_ext, r_ext


def extend_spherical_scalar(field, scales=1, reverse_theta=True):
    """Extend scalar field data to include boundary values."""
    basis = field.domain.bases[0]
    if not isinstance(basis, BallBasis):
        raise NotImplementedError("Shell not yet implemented.")
    # Fix scales
    if not isinstance(scales, (list, tuple, np.ndarray)):
        scales = (scales,) * 3
    # Build extended data
    field.require_scales(scales)
    shape = np.array(field['g'].shape) + np.array([1, 2, 2])
    data_ext = np.zeros(shape=shape, dtype=field.dtype)
    # Copy interior data
    data_ext[:-1, 1:-1, 1:-1] = field['g']
    # Interpolate to outer radius
    field_outer = field(r=basis.radial_basis.radius).evaluate()
    field_outer.require_scales(scales)
    data_ext[:-1, 1:-1, -1:] = field_outer['g']
    # Duplicate phi = 0 for phi = 2π
    data_ext[-1, :, :] = data_ext[0, :, :]
    # Average around poles
    data_ext[:, 0, :] = np.mean(data_ext[:, 1, :], axis=0)[None, :]
    data_ext[:, -1, :] = np.mean(data_ext[:, -2, :], axis=0)[None, :]
    # Average around origin
    data_ext[:, :, 0] = np.mean(data_ext[:, :, 1])[None, None]
    # Reverse theta
    if reverse_theta:
        data_ext = data_ext[:, ::-1, :]
    return data_ext


def build_spherical_interpolator(*fields, scales=1, bounds_error=False, **kw):
    """Build low-order interpolator for scalar fields in spherical coordinates."""
    basis = fields[0].domain.bases[0]
    coords_ext = extend_spherical_coords(basis, scales)
    coords_ext_flat = [c.ravel() for c in coords_ext]
    fields_ext = [extend_spherical_scalar(f, scales) for f in fields]
    fields_ext = np.squeeze(np.stack(fields_ext, axis=-1))
    return RegularGridInterpolator(coords_ext_flat, fields_ext, bounds_error=bounds_error, **kw)


def build_cartesian_interpolator(*args, **kw):
    spherical_interpolator = build_spherical_interpolator(*args, **kw)
    return lambda p: spherical_interpolator(cartesian_to_spherical(p))


def cartesian_to_spherical(p):
    """Convert [[x, y, z]] to [[phi, theta, r]] point sets."""
    p = np.array(p)
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]
    r = (x**2 + y**2 + z**2) ** 0.5
    phi = np.arctan2(y, x) % (2 * np.pi)
    theta = np.arccos(z / r)
    q = np.zeros_like(p)
    q[..., 0] = phi
    q[..., 1] = theta
    q[..., 2] = r
    return q


def create_integration_weigths(fields, dtype, radius=1):
    b = fields[0]
    d = fields[1]
    p = field.Field(dist=d, bases=(b,), dtype=dtype)
    weight_theta = b.local_colatitude_weights(b.domain.dealias[1])
    weight_r = b.local_radial_weights(b.domain.dealias[2])
    reducer = GlobalArrayReducer(d.comm_cart)
    p.require_scales(p.domain.dealias)
    int_norm = np.sum(weight_r * weight_theta * (0 * p['g'] + 1))
    int_norm = reducer.reduce_scalar(int_norm, MPI.SUM)
    int_correction = 4 / 3 * np.pi * radius**3 / int_norm
    return (weight_r, weight_theta, int_correction), reducer


def integrate_scalar(weights, reducer, field):
    # weights = (weights_r, weights_theta,  int_correction)
    field.require_scales(field.domain.dealias)
    field_int = np.sum(weights[0] * weights[1] * field['g'])
    field_int = reducer.reduce_scalar(field_int, MPI.SUM)
    return field_int * weights[2]
