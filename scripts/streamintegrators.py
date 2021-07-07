import numpy as np
import numba as nb
from numba import jit,prange
#import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
#from dedalus.extras.flow_tools import GlobalArrayReducer
import sphere_processing as sphere
#from mpi4py import MPI
#from joblib import Parallel, delayed
#import multiprocessing
import time




@jit(nopython=True,parallel=True)
def compute_link_DS(l,k):
    """ Computes the linking number between 2 polylines - uses JIT compiling via numba and auto-parallelization to get faster """
    λ = 0
    Nl = l.shape[0]
    Nk = k.shape[0]
    # Add first point to close loop
    ls0 = np.expand_dims(l[0,:], 0)
    ks0 = np.expand_dims(k[0,:], 0)
    ls = np.vstack((l, ls0))
    ks = np.vstack((k, ks0))
    for i in prange(Nk):
        for j in range(Nl):
            a = ls[j,:] - ks[i,:]
            b = ls[j,:] - ks[i+1,:]
            c = ls[j+1,:] - ks[i+1,:]
            d = ls[j+1,:] - ks[i,:]
            
            p = np.dot(a, np.cross(b,c))
            an = np.sqrt(np.dot(a,a))
            bn = np.sqrt(np.dot(b,b))
            cn = np.sqrt(np.dot(c,c))
            dn = np.sqrt(np.dot(d,d))
            
            d1 = an*bn*cn + np.dot(a,b) * cn + np.dot(b,c) *  an + np.dot(c,a) * bn
            d2 = an*dn*cn + np.dot(a,d) * cn + np.dot(d,c) *  an + np.dot(c,a) * dn
            
            λ += (np.arctan2(p, d1) + np.arctan2(p, d2))
    return λ/(2*np.pi)


def generate_streamlines(N_sample, num_cores, fun):
      # sample N_sample random points uniformly inside the ball (Polar sampling method)
    u = 2 * np.random.rand(N_sample) - 1
    p = 2*np.pi * np.random.rand(N_sample)
    r = np.random.rand(N_sample) ** (1/3.)

    x_0 = r * np.cos(p) * np.sqrt(1-u*u)
    y_0 = r * np.sin(p) * np.sqrt(1-u*u)
    z_0 = r * u

    points = [np.array((x_0[i],y_0[i],z_0[i])) for i in range(N_sample)]
    #
    #streamlines = Parallel(n_jobs=num_cores)(delayed(fun)(x) for x in points)

    streamlines = []
    for x in points:
        streamlines.append(fun(x))
    return streamlines



class StreamGen:

    def __init__(self, num_cores, N_rounds, N_sample, T, ug):
        self.num_cores = num_cores
        self.N_rounds = N_rounds
        self.N_sample = N_sample
        self.hel_est = 0

        self.T = T # streamline integration time


        shape = ug.shape[1:]
        dtype = ug.dtype
        c, d, b = sphere.build_ball(shape, dtype=dtype)
        u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
        u['g'] = ug

        self.u = u # vel. field

        curl = operators.Curl
        integral = operators.integrate
        dot = arithmetic.DotProduct
        self.ω = curl(u)
        ex, ey, ez = sphere.build_cartesian_unit_vectors(b)
        self.ωx = dot(ex, self.ω).evaluate()
        self.ωy = dot(ey, self.ω).evaluate()
        self.ωz = dot(ez, self.ω).evaluate()

        # Parameters
        interp_scales = 4 # Controls refinement of spectral sampling prior to linear interpolation

        # Build cartesian vorticity interpolator
        self.ω_interp = sphere.build_cartesian_interpolator(self.ωx, self.ωy, self.ωz, scales=interp_scales)

        # Helicity operator: integrate over the ball to obtain the helicity of the field
        self.H_op = dot(u, self.ω)



    def wrap_ω_interp(self, t, x):
        return self.ω_interp(x)

    def integrate_streamline(self, x0):
        RHS = self.wrap_ω_interp
        t_bounds = [0, self.T]
        rtol = 1e-3
        atol = 1e-4
        sol = solve_ivp(self.wrap_ω_interp, t_bounds, x0, rtol=rtol, atol=atol, dense_output=True, max_step=1e-2)
        return sol

    def generate_streamlines_method(self):
        # sample N_sample random points uniformly inside the ball (Polar sampling method)
        u = 2 * np.random.rand(N_sample) - 1
        p = 2*np.pi * np.random.rand(N_sample)
        r = np.random.rand(N_sample) ** (1/3.)

        x_0 = r * np.cos(p) * np.sqrt(1-u*u)
        y_0 = r * np.sin(p) * np.sqrt(1-u*u)
        z_0 = r * u

        points = [np.array((x_0[i],y_0[i],z_0[i])) for i in range(N_sample)]

        #
        streamlines = Parallel(n_jobs=self.num_cores)(delayed(self.integrate_streamline)(x) for x in points)



        return streamlines


    def estimate_helicity(self):

        hel_estimates_seq = [0]
        sample_times = np.linspace(0,self.T, 100)
        for _ in range(self.N_rounds):
            #streamlines = self.generate_streamlines_method()
            streamlines = generate_streamlines(self.N_sample, self.num_cores, self.integrate_streamline)
            discrete_stream = []
            for i in range(N_sample):
                discrete_stream.append(streamlines[i].sol(sample_times).T)
            for i in range(N_sample):
                l = discrete_stream[i]
                print(l.shape)
                for j in range(0, i-1):
                    #discretize streamlines
                    self.hel_est += 2 * compute_link_DS(l,discrete_stream[j])
                    self.hel_est /= self.T*self.T
                    hel_estimates_seq.append(self.hel_est)
        return hel_estimates_seq
