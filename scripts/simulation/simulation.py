

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.tools import logging
import time
from mpi4py import MPI
from gns_parameters import scales_to_Gammas

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'


# Parameters
n = 128
radius = 1
dealias = 3 / 2
kc = 8 * np.pi
kw = 4
sc = 1
timestep = 1 / 20
stop_iteration = np.int(np.round(1000 / timestep)) + 1
dtype = np.float64
timestepper = timesteppers.RK443
plot_matrices = False
scalar_cadence = int(np.round(0.1 / timestep))
snapshot_cadence = int(np.round(100 / timestep))
noise_amp = 1e-3
noise_scale = 0.5
ncc_cutoff = 1e-6
entry_cutoff = 1e-8
mesh = [8, 24]

# Derived parameters
Nphi = n
Ntheta = n // 2
Nr = n // 2
gamma0, gamma2, gamma4 = scales_to_Gammas(kc, kw, sc)
logger.info("Gammas: %0.3e, %0.3e, %0.3e" %(gamma0, gamma2, gamma4))

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (Nphi, Ntheta-1, Nr), radius=radius, dealias=(dealias,dealias,dealias), dtype=dtype)
bo = b._new_k(2)
bs = b.S2_basis()
br = b.radial_basis
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
p = field.Field(dist=d, bases=(b,), dtype=dtype)
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
t1 = field.Field(dist=d, bases=(bs,), tensorsig=(c,), dtype=dtype)
t2 = field.Field(dist=d, bases=(bs,), tensorsig=(c,), dtype=dtype)
t3 = field.Field(dist=d, bases=(bs,), tensorsig=(c,), dtype=dtype)
R = field.Field(dist=d, bases=(br,), tensorsig=(c,), dtype=dtype)
R['g'][2] = r

# Operators
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
curl = lambda A: operators.Curl(A)
dot = lambda A, B: arithmetic.DotProduct(A, B)
dt = lambda A: operators.TimeDerivative(A)
LiftTau = lambda A, n: operators.LiftTau(A, bo, n)
transpose = lambda A: operators.TransposeComponents(A)
rad = lambda A: operators.RadialComponent(A)
ang = lambda A: operators.AngularComponent(A)
id = lambda A: A

# Problem
e = grad(u) + transpose(grad(u))
sigma0 = gamma0 * e
sigma2 = - gamma2 * lap(e)
sigma4 = gamma4 * lap(lap(e))
sigma = sigma0 + sigma2 + sigma4
div_taus = dot(R, LiftTau(t2, -1)) + dot(R, LiftTau(t3, -2))
mom_taus = LiftTau(t1, -1) + LiftTau(t2, -2) + LiftTau(t3, -3)

problem = problems.IVP([p, u, t1, t2, t3])
problem.add_equation((div(u) + div_taus, 0), condition="ntheta != 0")
problem.add_equation((dt(u) - div(sigma) + grad(p) + mom_taus, -dot(u,grad(u))), condition="ntheta != 0")
problem.add_equation((u(r=radius), 0), condition="ntheta != 0")
problem.add_equation((rad(grad(u)(r=radius)), 0), condition="ntheta != 0")
problem.add_equation((rad(rad(grad(grad(u))(r=radius))), 0), condition="ntheta != 0")
problem.add_equation((p, 0), condition="ntheta == 0")
problem.add_equation((u, 0), condition="ntheta == 0")
problem.add_equation((t1, 0), condition="ntheta == 0")
problem.add_equation((t2, 0), condition="ntheta == 0")
problem.add_equation((t3, 0), condition="ntheta == 0")

# Solver
solver = solvers.InitialValueSolver(problem, timestepper, ncc_cutoff=ncc_cutoff, entry_cutoff=entry_cutoff)
solver.stop_iteration = stop_iteration

# Initial conditions
u.set_scales(noise_scale)
u['g'] = noise_amp * np.random.randn(*u['g'].shape)
u.require_scales(1)

# Plot matrices
if plot_matrices:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for sp in solver.subproblems:
        ell = sp.group[1]
        A = sp.M_min + timestep*sp.L_min
        plt.imshow(np.log10(np.abs((A @ sp.pre_right).A)), interpolation='nearest')
        plt.colorbar()
        plt.savefig('matrices/ell_%i.png' %ell, dpi=300)
        plt.clf()
        print('ell = %i, cond = %.2e' %(ell, np.linalg.cond(A.A)))
    raise

# Analysis
weight_theta = b.local_colatitude_weights(dealias)
weight_r = b.local_radial_weights(dealias)
reducer = GlobalArrayReducer(d.comm_cart)
p.require_scales(p.domain.dealias)
int_norm = np.sum(weight_r * weight_theta * (0*p['g'] + 1))
int_norm = reducer.reduce_scalar(int_norm, MPI.SUM)
int_correction = 4 / 3 * np.pi * radius**3 / int_norm

def integrate_scalar(field):
    field.require_scales(field.domain.dealias)
    field_int = np.sum(weight_r * weight_theta * field['g'])
    field_int = reducer.reduce_scalar(field_int, MPI.SUM)
    return field_int * int_correction

t_list = []
E_list = []
H_list = []
E_op = 0.5 * dot(u, u)
H_op = dot(u, curl(u))

snapshots = solver.evaluator.add_file_handler('snapshots', iter=snapshot_cadence, virtual_file=True, max_writes=1)
snapshots.add_task(p, name='p')
snapshots.add_task(u, name='u')
snapshots.add_task(t1, name='t1')
snapshots.add_task(t2, name='t2')
snapshots.add_task(t3, name='t3')
snapshots.add_task(E_op, name='E')
snapshots.add_task(H_op, name='H')

# Main loop
start_time = time.time()
while solver.ok:
    if solver.iteration % scalar_cadence == 0:
        ti = solver.sim_time
        Ei = integrate_scalar(E_op.evaluate())
        Hi = integrate_scalar(H_op.evaluate())
        logger.info("t = %f, E = %e, H = %e" %(ti, Ei, Hi))
        t_list.append(ti)
        E_list.append(Ei)
        H_list.append(Hi)
    solver.step(timestep)
end_time = time.time()
logger.info('Run time: %.2i' %(end_time-start_time))
np.savez('scalars.npz', t=t_list, E=E_list, H=H_list)
