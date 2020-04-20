import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev
import matplotlib.pyplot as plt


def hessian(fun):
    return jit(jacfwd(jacrev(fun)))


def bumpy_paraboloid(x):
    """An example function to minimize."""
    y = x[0]**2 + jnp.abs(x[0]) * jnp.cos(x[0])
    y += x[1]**2
    return y


def newton_min(f, x0, max_n_steps=20, gamma=1., tol=1e-6):
    """Do minimization using Newton's method."""
    x_history = [x0]
    grad_f = grad(f)
    grad2_f = hessian(f)
    for k in range(max_n_steps):
        grad_f_x = grad_f(x_history[k])
        if grad_f_x @ grad_f_x < tol:
            break
        step = gamma * np.linalg.pinv(grad2_f(x_history[k])) @ grad_f_x
        x_history.append(
            x_history[k] - step)
    return x_history


def main():
    x1 = np.linspace(-5, 5)
    x2 = np.linspace(-5, 5)
    Y = np.zeros((len(x1), len(x2)))
    for i1 in range(len(x1)):
        for i2 in range(len(x2)):
            Y[i1, i2] = bumpy_paraboloid([x1[i1], x2[i2]])

    fig, ax = plt.subplots()
    contourset = ax.contour(
        x1, x2, Y,
        levels=[1., 2., 4., 8., 16., 32.])
    ax.clabel(contourset, inline=1, fontsize=10, fmt='%.0f')
    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$')

    x_history = newton_min(
        bumpy_paraboloid, x0=jnp.array([4., 4.]), gamma=0.5)

    x_history = np.array(x_history)
    print(x_history)
    ax.plot(
        x_history[:, 0], x_history[:, 1],
        color='black', marker='o')

    plt.show()

if __name__ == '__main__':
    main()
