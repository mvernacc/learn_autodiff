import numpy as np

def conj_grad(A, b, x0, n_steps=None, tol=1e-9):
    """Conjugate gradient method to solve A x = b.
    See https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    """
    if n_steps is None:
        n_steps = len(b)
    x = x0
    r = b - A @ x0
    r2_old = r @ r
    p = r
    for i in range(n_steps):
        Ap = A @ p
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

def main():
    A = np.array([[4., 1.], [1., 3.]])
    b = np.array([1., 2])
    x = conj_grad(A, b, x0=np.array([2., 1.]))
    print(x)


if __name__ == '__main__':
    main()
