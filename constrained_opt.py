"""Practice constrained optimization with jax."""

import jax
import jax.numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# These functions are for the first example from
# https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization

def rosen(x, a=1, b=100):
    """Rosenbrock function"""
    return (a - x[0])**2  + b * (x[1] - x[0]**2)**2


def constraints_fun(x):
    c_line = x[0] + x[1] - 2
    c_cubic = (x[0] - 1)**3 - x[1] + 1
    return np.array([c_line, c_cubic])


def main():
    constraint = NonlinearConstraint(
        fun=constraints_fun, lb=-np.inf, ub=0,
        jac=jax.jacfwd(constraints_fun))

    objective = rosen

    # From (1.5, 1.5), 'trust-constr' finds the global minimum (1, 1)
    x_guess = np.array([1.5, 1.5])
    # From (-1, 1), 'trust-constr' fins the local min at (0, 0)
    # x_guess = np.array([-1., 1.])

    # Hessian vector product function for the objective
    # This returns a vector of length n, where n is len(x)
    hessp = lambda x, v: jax.grad(
        lambda x: np.vdot(jax.grad(objective)(x), v))(x)

    result = minimize(
        fun=objective, x0=x_guess,
        method='trust-constr',
        jac=jax.grad(objective),
        constraints=constraint, hessp=hessp)

    print(result)


if __name__ == '__main__':
    main()
