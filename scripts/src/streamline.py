import numpy as np
from scipy.integrate import solve_ivp, quad

class Streamline():
    """Streamline integrator.

    Parameters
    ----------
    x0 : ndarray with shape (3,)
        Initial cartesian position.
    L : float
        Arclength for integration.
    rtol, atol, max_Step : float
        Parameters passed to scipy.integrate.solve_ivp.

    Notes
    -----
    The streamline is integrated in arclength coordinates by normalizing the tangent vector
    field supplied to the integrate method.
    """

    def __init__(self, x0, L, rtol, atol, max_step):
        self.x0 = x0
        self.L = L
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step

    def integrate(self, vector_interpolator):
        dy = np.zeros(5)
        # Define integration RHS
        def RHS(s, y):
            vector = vector_interpolator(y[0:3])[0]
            vector_mag = np.linalg.norm(vector)
            np.divide(vector, vector_mag, out=dy[0:3])
            dy[3] = vector_mag
            dy[4] = 1 / vector_mag
            return dy
        # Setup extended initial conditions
        y0 = np.append(self.x0, [0, 0])
        # Integrate
        self.sol = sol = solve_ivp(RHS, (0, self.L), y0, rtol=self.rtol, atol=self.atol,
                                   dense_output=True, max_step=self.max_step)
        if sol.status != 0:
            raise ValueError('solve_ivp did not converge')
        # References
        self.s = sol.t
        self.x = sol.y[0:3].T
        self.C = sol.y[3]
        self.T = sol.y[4]
        self.C_final = self.C[-1]
        self.T_final = self.T[-1]

