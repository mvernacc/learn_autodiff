import jax.numpy as jnp
from jax import grad, jit, vmap
import matplotlib.pyplot as plt

def poly(x):
    return x**3 - 2 * x**2 + 1

def main():
    grad_poly = grad(poly)
    grad2_poly = grad(grad_poly)

    x = jnp.linspace(-4, 4, 100)

    plt.plot(x, poly(x), label='$f$')
    plt.plot(x, vmap(grad_poly)(x), label="$f'$")
    plt.plot(x, vmap(grad2_poly)(x), label="$f''$")
    plt.xlabel('$x$')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
