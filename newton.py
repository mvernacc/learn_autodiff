import numpy as np_orig
import jax.numpy as np
from jax import grad, jit, vmap, jacfwd, jacrev
import matplotlib.pyplot as plt


def hessian(fun):
    return jit(jacfwd(jacrev(fun)))


def bumpy_paraboloid(x):
    """An example function to minimize."""
    y = x[0]**2 + 0.125 * np.cos(4 * x[0])
    y += x[1]**2
    return y


def newton_min(f, x0, max_n_steps=20, gamma=1., lamb=0.1, tol=1e-6):
    """Do minimization using Newton's method."""
    x_history = [x0]
    grad_f = grad(f)
    grad2_f = hessian(f)
    for k in range(max_n_steps):
        grad_f_x = grad_f(x_history[k])
        if grad_f_x @ grad_f_x < tol:
            break
        B = np_orig.linalg.pinv(grad2_f(x_history[k])) + lamb * np.eye(len(x0))
        step = gamma * B @ grad_f_x
        x_history.append(
            x_history[k] - step)
    return x_history


def hvp(f, x, v):
    """Hessian vector product."""
    return grad(lambda x: np.vdot(grad(f)(x), v))(x)


def conj_grad_with_hvp(A_prod, b, x0, n_steps=None, tol=1e-9):
    """Conjugate gradient method to solve A x = b.
    A_prod is a function which maps x --> A x
    See https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    """
    if n_steps is None:
        n_steps = len(b)
    x = x0
    r = b - A_prod(x0)
    r2_old = r @ r
    p = r
    for i in range(n_steps):
        Ap = A_prod(p)
        alpha = r2_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r2_new = r @ r
        if r2_new < tol:
            break
        beta = r2_new / r2_old
        p = r + beta * p
        r2_old = r2_new
    return x


def trunc_newton_min(f, x0, max_n_steps=20, lamb=0.1, tol=1e-6,
        n_steps_inner=None):
    """Do minimization using Heassian-free truncated Newton's method.
    This only uses Hessian vector products (1 dimensional),
    not the full hessian (2 dimensional).
    For large `len(x)`, computing the full hessian would be slow and might not fit in memory.
    See http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf"""
    x_history = [x0]
    grad_f = grad(f)
    step = np.zeros(len(x0))
    for k in range(max_n_steps):
        x = x_history[k]
        grad_f_x = grad_f(x)
        if grad_f_x @ grad_f_x < tol:
            break
        # Inner conjugate-gradient solve for the step
        step = conj_grad_with_hvp(
            A_prod=lambda v: hvp(f, x, v) + lamb * v,
            b=-1 * grad_f_x,
            x0=step, n_steps=n_steps_inner)
        x_history.append(
            x_history[k] + step)
    return x_history


def main():
    ### Make a contour plot of the function that we're minimizing ###
    x1 = np_orig.linspace(-5, 5)
    x2 = np_orig.linspace(-5, 5)
    Y = bumpy_paraboloid(np_orig.meshgrid(x1, x2))

    fig, ax = plt.subplots()
    contourset = ax.contour(
        x1, x2, Y,
        levels=[1., 2., 4., 8., 16., 32.])
    ax.clabel(contourset, inline=1, fontsize=10, fmt='%.0f')
    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$')

    ### Do the minimization ###
    # x_history = newton_min(
    #     bumpy_paraboloid, x0=np.array([4., 4.]),
    #     lamb=0.05, gamma=0.25)
    x_history = trunc_newton_min(
        bumpy_paraboloid, x0=np.array([4., 4.]),
        lamb=0.05)
    x_history_ni1 = trunc_newton_min(
        bumpy_paraboloid, x0=np.array([4., 4.]),
        lamb=0.05, n_steps_inner=1)

    ### Plot the history of x's the minimizer steped thru ###
    x_history = np_orig.array(x_history)
    print(x_history)
    x_history_ni1 = np_orig.array(x_history_ni1)
    print(x_history_ni1)

    ax.plot(
        x_history[:, 0], x_history[:, 1],
        color='black', marker='.', label='Inner CG to completion')
    ax.plot(
        x_history_ni1[:, 0], x_history_ni1[:, 1],
        color='grey', marker='.', label='Inner CG only 1 step')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
