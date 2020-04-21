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
    x_history = newton_min(
        bumpy_paraboloid, x0=np.array([4., 4.]), gamma=0.05)

    ### Plot the history of x's the minimizer steped thru ###
    x_history = np_orig.array(x_history)
    print(x_history)
    ax.plot(
        x_history[:, 0], x_history[:, 1],
        color='black', marker='o')

    plt.show()

if __name__ == '__main__':
    main()
