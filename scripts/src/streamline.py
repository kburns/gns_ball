import numpy as np
from scipy.integrate import solve_ivp, quad

class Streamlines():

    def __init__(self, x0, t_bounds, tols, interpolator, normalized=False, max_step=1e-3):
        self.x0 = x0
        self.t_bounds = t_bounds
        self.rtol = tols[0]
        self.atol = tols[1]
        self.max_step = max_step
        self.interpolator = interpolator
        self.normalized = normalized

    def wrap_interp(self, t, x):
        return self.interpolator(x)

    def wrap_interp_normalized(self, t, x):
        ωi = self.interpolator(x)
        ω_mag = np.linalg.norm(ωi)
        return ωi / ω_mag

    def integrate(self):
        # Define integrator function: if normalized
        if self.normalized:
            self.RHS = self.wrap_interp_normalized
        else:
            self.RHS = self.wrap_interp
        # Integrate streamline
        sol = solve_ivp(self.RHS,
                        self.t_bounds,
                        self.x0,
                        rtol=self.rtol,
                        atol=self.atol,
                        dense_output=True,
                        max_step=self.max_step)
        self.status = sol.status
        if self.status < 0:
            raise ValueError('Integrator not converged')
        if self.normalized:
            # integrate circulation  C = \int_0^t_max np.dot(omega(r(t)), dr(t)), with dr = RHS dt
            # Note that in the normalized case, t has units of length.

            def circulation(t):
                return np.dot(self.interpolator(sol.sol(t)), self.RHS(1., sol.sol(t)).T)
            C = quad(circulation, self.t_bounds[0], self.t_bounds[1], limit=500)
            L = self.t_bounds[1] - self.t_bounds[0]  # Arc length of streamline: exactly known

            self.T = L / C[0]

        else:  # unnormalized case: t has units of time.
            self.T = self.t_bounds[1] - self.t_bounds[0]
        self.sol = sol
