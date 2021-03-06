from typing import Callable, Union, Tuple
import jax
import jax.interpreters.xla._DeviceArray as JaxArray
import numpy as np
from scipy.optimize import line_search
import matplotlib.pyplot as plt


XMIN = -2
XMAX = 2
YMIN = -1
YMAX = 3
CONTOUR_X = np.linspace(XMIN, XMAX)
CONTOUR_Y = np.linspace(YMIN, YMAX)


def rosen(x, a=1, b=5):
    """Rosenbrock function"""
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


def draw_contours(ax, f, color='black'):
    contour_z = np.zeros((len(CONTOUR_X), len(CONTOUR_Y)))
    for ix, x in enumerate(CONTOUR_X):
        for iy, y in enumerate(CONTOUR_Y):
            contour_z[iy, ix] = f(np.array([x, y]))

    ax.contour(
        CONTOUR_X, CONTOUR_Y, contour_z,
        levels=np.logspace(-1, 3, 8), colors=color)


def do_bfgs_step(f: Callable, f_grad: Callable,
                 x_k: Union[np.ndarray, JaxArray], H_k: Union[np.ndarray, JaxArray]
                 ) -> Tuple[Union[np.ndarray, JaxArray], Union[np.ndarray, JaxArray]]:
    """Do one update step of the BFGS quasi-Newton optimization algorithm.

    See Noedcal and Wright 2006, Chapter 6.1

    Args:
        f (Callable): Objective function.
        f_grad (Callable): Function giving the gradient of the objective.
        x_k (Union[np.ndarray, JaxArray]): Starting point for this step.
        H_k (Union[np.ndarray, JaxArray]): Inverse hessian estimate at start of this step.

    Returns:
        Tuple[Union[np.ndarray, JaxArray], Union[np.ndarray, JaxArray]]:
            x_k_plus_1: Next point.
            H_k_plus_1: Next inverse hessian estimate.
    """
    grad_f_k = f_grad(x_k)

    # Compute search direction
    p_k = - H_k @ grad_f_k

    # Update x
    alpha_k, _, _, _, _, grad_f_k_plus_1 = line_search(
        f, f_grad, x_k, p_k, grad_f_k)
    print(f'alpha_k = {alpha_k:.4f}')
    x_k_plus_1 = x_k + alpha_k * p_k

    # Update the approximation of the inverse Hessian.
    s_k = x_k_plus_1 - x_k
    y_k = grad_f_k_plus_1 - grad_f_k
    # Equation 6.14
    rho_k = 1 / (y_k.transpose() @ s_k)
    I = np.eye(len(x_k))
    # BFGS upadate -- Equation 6.17
    H_k_plus_1 = ((I - rho_k * s_k @ y_k.transpose()) @ H_k @ (I - rho_k * y_k @ s_k.transpose())
                  + rho_k * s_k @ s_k.transpose())
    return x_k_plus_1, H_k_plus_1


def draw_bfgs_step(ax, f, f_grad, x_k, x_k_plus_1, B_k):
    # Draw contours of the true function.
    draw_contours(ax, f, color='black')

    # Draw contours of the BFGS quadratic approximation.
    f_k = f(x_k)
    grad_f_k = f_grad(x_k)

    def bfgs_quad_approx(x):
        p = x - x_k
        return f_k + grad_f_k @ p + 0.5 * p.transpose() @ B_k @ p

    draw_contours(ax, bfgs_quad_approx, color='tab:blue')

    # Draw the step to the next x point.
    step = x_k_plus_1 - x_k
    ax.arrow(
        x_k[0], x_k[1], step[0], step[1],
        fc='tab:red', ec='tab:red',
        head_width=0.05, head_length=0.1)


def main():
    n_steps = 5
    fig, axes = plt.subplots(ncols=n_steps, figsize=(20, 6))
    f = rosen
    f_grad = jax.grad(f)

    x_k = np.array([0., 2.])
    H_k = (1. / np.linalg.norm(f_grad(x_k))) * np.eye(2)

    for k in range(n_steps):
        print(f'\nStep k={k:d}')
        print('x_k = ' + str(x_k))
        print('H_k = ' + str(H_k))

        x_k_plus_1, H_k_plus_1 = do_bfgs_step(f, f_grad, x_k, H_k)

        # Plotting
        ax = axes[k]
        B_k = np.linalg.inv(H_k)
        draw_bfgs_step(ax, f, f_grad, x_k, x_k_plus_1, B_k)
        ax.set_title(f'$k = {k:d}$')

        x_k = x_k_plus_1
        H_k = H_k_plus_1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
