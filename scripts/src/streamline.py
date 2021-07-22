import numpy as np
from scipy.integrate import solve_ivp, quad

class Streamline():

    def __init__(self, x0, T, tols, interpolator, normalized=False, max_step=1e-3):
        self.x0 = x0
        self.t_bound = T
        self.rtol = tols[0]
        self.atol = tols[1]
        self.max_step = max_step
        self.interpolator = interpolator
        self.normalized = normalized

    def wrap_interp(self, t, x):
        return self.interpolator(x)

    def wrap_interp_normalized(self, t, x):
        ωi = self.interpolator(x[0:3])[0]
        #ωi = self.interpolator(x)
        ω_mag = np.linalg.norm(ωi)
        #return ωi / ω_mag
        return np.append(ωi / ω_mag, ω_mag)

    def integrate(self):
        # Define integrator function: if normalized
        if self.normalized:
            self.RHS = self.wrap_interp_normalized
            x0s = np.hstack((self.x0, 0.)) # integrate circulation simulteneaously
            #x0s = self.x0
        else:
            self.RHS = self.wrap_interp
            x0s = self.x0
            # Integrate streamline
        sol = solve_ivp(self.RHS,
                        (0, self.t_bound),
                        x0s,
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
            # = norm of interpolator

            C = sol.y[-1,:] # total circulation
            L = self.t_bound  # Arc length of streamline: exactly known
            self.C = C
            self.y = sol.y[0:3,:].T
            self.T = L / C[-1]

        else:  # unnormalized case: t has units of time.
            self.T = self.t_bound
            self.y = sol.y.T
        self.sol = sol
