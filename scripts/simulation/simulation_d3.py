# +
"""
Simulation code for  GNS parity-breaking flows in ball in dedalus 3
Outputs elements for plotting according to ivp_shell_convection dedalus 3 example.

"""

import numpy as np
import dedalus.public as d3
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.tools import logging
import time
from mpi4py import MPI
#from gns_parameters import scales_to_Gammas

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'

def scales_to_Gammas(k_center, k_width, growth_rate):
    kc = k_center
    kw = k_width
    s = growth_rate
    gamma2 = -2 * kc**2
    gamma0 = (kw**2 + gamma2)**2 / 4
    Gamma4 = 2 * s / gamma2 / (gamma0 - gamma2**2 / 4)
    Gamma0 = gamma0 * Gamma4
    Gamma2 = gamma2 * Gamma4
    return Gamma0, Gamma2, Gamma4

def simrun(replica_idx):
    # Parameters
    n = 128
    radius = 1
    dealias = 3 / 2
    kc = 8 * np.pi
    kw = 4
    sc = 1
    timestep = 1 / 20
    stop_iteration = int(np.round(200 / timestep)) + 1
    dtype = np.float64
    timestepper = timesteppers.RK443
    plot_matrices = False
    scalar_cadence = int(np.round(1 / timestep))
    #snapshot_cadence = int(np.round(100 / timestep))
    noise_amp = 1e-3
    noise_scale = 0.5
    ncc_cutoff = 1e-6
    entry_cutoff = 1e-8
    #mesh = [8, 24]
    
    # Derived parameters
    Nphi = n
    Ntheta = n // 2
    Nr = n // 2
    gamma0, gamma2, gamma4 = scales_to_Gammas(kc, kw, sc)
    logger.info("Gammas: %0.3e, %0.3e, %0.3e" %(gamma0, gamma2, gamma4))
    
    # Bases
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    #d = distributor.Distributor((c,), mesh=mesh)
    d = d3.Distributor(c, dtype=dtype)
    ball = d3.BallBasis(c, shape=(Nphi, Ntheta, Nr), radius=radius, dealias=dealias, dtype=dtype)
    bo = ball._new_k(2)
    bs = ball.S2_basis(radius=radius)
    br = ball.radial_basis
    #bs = ball.surface

    phi, theta, r = d.local_grids(ball)
    
    # Fields
    p = d.Field(bases=ball, name='p')
    u = d.VectorField(c,  bases=ball, name='u')
    tau_1 = d.VectorField(c, bases=bs, name='tau_1')
    tau_2 = d.VectorField(c, bases=bs, name='tau_2')
    tau_3 = d.VectorField(c, bases=bs, name='tau_3')
    #t3 = d.Field(c, bases=bs, dtype=dtype, name='t3')
    R = d.VectorField(c, bases=br, dtype=dtype, name='R')
    R['g'][2] = r
    
    # Operators
    # div = lambda A: operators.Divergence(A, index=0)
    # lap = lambda A: operators.Laplacian(A, c)
    # grad = lambda A: operators.Gradient(A, c)
    # curl = lambda A: operators.Curl(A)
    dot = lambda A, B: arithmetic.DotProduct(A, B)
    dt = lambda A: operators.TimeDerivative(A)
    LiftTau = lambda A, n: operators.Lift(A, bo, n)
    #transpose = lambda A: operators.TransposeComponents(A)
    rad = lambda A: operators.RadialComponent(A)
    #ang = lambda A: operators.AngularComponent(A)
    #id = lambda A: A
    
    # Problem
    e = d3.grad(u) + d3.trans(d3.grad(u))
    sigma0 = gamma0 * e
    sigma2 = - gamma2 * d3.lap(e)
    sigma4 = gamma4 * d3.lap(d3.lap(e))
    sigma = sigma0 + sigma2 + sigma4
    string_sigma = " - div(gamma0 * e - gamma2* lap(e) + gamma4*lap(lap(e))) "
    div_taus = d3.dot(R, LiftTau(tau_2, -1)) + d3.dot(R, LiftTau(tau_3, -2))
    mom_taus = LiftTau(tau_1, -1) + LiftTau(tau_2, -2) + LiftTau(tau_3, -3)
    #lift = lambda A: d3.Lift(A, ball, -1)

    problem = d3.IVP([p, u, tau_1, tau_2,  tau_3],  namespace=locals())
    #problem.add_equation("div(u) + div_taus = 0", condition="ntheta != 0")
    problem.add_equation((d3.div(u) + div_taus, 0), condition="ntheta != 0")
    #problem.add_equation("dt(u) " + string_sigma + " + grad(p) + mom_taus =  - dot(u, grad(u))", condition="ntheta != 0")
    problem.add_equation((dt(u) - d3.div(sigma) + d3.grad(p) + mom_taus, -dot(u,d3.grad(u))), condition="ntheta != 0")

    #problem.add_equation("radial(u(r=radius)) = 0", condition="ntheta != 0") # no penetration
    problem.add_equation((u(r=radius), 0), condition="ntheta != 0")
    #problem.add_equation("radial(grad(u)(r=radius)) = 0", condition="ntheta != 0")
    #problem.add_equation("radial(radial(grad(grad(u))(r=radius))) = 0", condition="ntheta != 0")
    problem.add_equation((rad(d3.grad(u)(r=radius)), 0), condition="ntheta != 0")
    problem.add_equation((rad(rad(d3.grad(d3.grad(u))(r=radius))), 0), condition="ntheta != 0")

    #problem.add_equation("integ(p) = 0")  # Pressure gauge
    problem.add_equation((p, 0), condition="ntheta == 0")
    problem.add_equation((u, 0), condition="ntheta == 0")
    problem.add_equation((tau_1, 0), condition="ntheta == 0")
    problem.add_equation((tau_2, 0), condition="ntheta == 0")
    problem.add_equation((tau_3, 0), condition="ntheta == 0")
    
    # Solver
    solver = solvers.InitialValueSolver(problem, timestepper, ncc_cutoff=ncc_cutoff, entry_cutoff=entry_cutoff)
    solver.stop_iteration = stop_iteration
    
    # Initial conditions
    #u.set_scales(noise_scale)
    #u['g'] = noise_amp * np.random.randn(*u['g'].shape)
    #u.require_scales(1)
    
    u.fill_random('g', distribution='normal', scale=noise_amp)
    #u['g'] *= noise_amp
    
    # Analysis
    #weight_theta = b.local_colatitude_weights(dealias)
    #weight_r = b.local_radial_weights(dealias)
    #reducer = GlobalArrayReducer(d.comm_cart)
    #p.require_scales(p.domain.dealias)
    #int_norm = np.sum(weight_r * weight_theta * (0*p['g'] + 1))
    #int_norm = reducer.reduce_scalar(int_norm, MPI.SUM)
    #int_correction = 4 / 3 * np.pi * radius**3 / int_norm
    
    
    
    #def integrate_scalar(field):
    #    #field.require_scales(field.domain.dealias)
    #    field_int = np.sum(weight_r * weight_theta * field['g'])
    #    field_int = reducer.reduce_scalar(field_int, MPI.SUM)
    #    return field_int * int_correction
    
    t_list = []
    E_list = []
    H_list = []
    E_op = 0.5 * d3.dot(u, u)
    H_op = d3.dot(u, d3.curl(u))
    vort = 0.5 * d3.dot(d3.curl(u),d3.curl(u))
    
    slices = solver.evaluator.add_file_handler('slices{0}'.format(replica_idx), iter=scalar_cadence, max_writes=10)
    slices.add_task(H_op(phi=0), scales=dealias, name='H(phi=0)')
    slices.add_task(H_op(phi=np.pi), scales=dealias, name='H(phi=pi)')
    slices.add_task(H_op(phi=3/2*np.pi), scales=dealias, name='H(phi=3/2*pi)')
    slices.add_task(H_op(r=1), scales=dealias, name='H(r=1)')
    slices.add_task(vort(phi=0), scales=dealias, name='Ens(phi=0)')
    slices.add_task(vort(phi=np.pi), scales=dealias, name='Ens(phi=pi)')
    slices.add_task(vort(phi=3/2*np.pi), scales=dealias, name='Ens(phi=3/2*pi)')
    slices.add_task(vort(r=1), scales=dealias, name='Ens(r=1)')
    
    snapshots = solver.evaluator.add_file_handler('snapshots', iter=snapshot_cadence, virtual_file=True, max_writes=1)
    #snapshots.add_task(p, name='p')
    snapshots.add_task(u, name='u')
    #snapshots.add_task(t1, name='t1')
    #snapshots.add_task(t2, name='t2')
    #snapshots.add_task(t3, name='t3')
    #snapshots.add_task(E_op, name='E')
    #snapshots.add_task(H_op, name='H')
    
    flow = d3.GlobalFlowProperty(solver, cadence=scalar_cadence)
    #flow.add_property(d3.ave(E_op), name='E')
    flow.add_property(d3.dot(u,u), name='E')
    
    # Main loop
    start_time = time.time()
    while solver.proceed:
        if (solver.iteration-1) % scalar_cadence == 0:
            ti = solver.sim_time
            #Ei = flow.max('E')
            Ei = flow.max('E')
            logger.info("t = %f, E = %e" %(ti, Ei))
            t_list.append(ti)
            E_list.append(Ei)
            #H_list.append(Hi)
        solver.step(timestep)
    end_time = time.time()
    logger.info('Run time: %.2i' %(end_time-start_time))
    #np.savez('replicas/scalars{0}.npz'.format(replica_idx), t=t_list, E=E_list, H=H_list)

if __name__ == '__main__':
    N_replicas = 1
    for i in range(N_replicas):
        logger.info("--- Starting replica {0} for plotting ---".format(i))
        simrun(i)
        logger.info("--- End replica {0} for plotting ---".format(i))
    logger.info("===== All done =====")
# -













