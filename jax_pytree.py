"""Mini benchmarks of jit'ing a function which takes a user-defined class as an
argument."""
import jax
import jax.numpy as jnp
import numpy as onp
import timeit


@jax.tree_util.register_pytree_node_class
class Squarer:
    def __init__(self, coef):
        self.coef = coef

    def square(self, x):
        return self.coef * x**2

    def tree_flatten(self):
        return ((self.coef,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def f(sq: Squarer, x):
    return sq.square(x)


f_jitted = jax.jit(f)


def f_no_class(x):
    """Implements the same math as Squarer.square, but without the class."""
    return 2 * x**2


f_no_class_jitted = jax.jit(f_no_class)


x_test = jnp.array([0., 1., 2., 3.])
x_test_onp = onp.array([0., 1., 2., 3.])
sq = Squarer(2.)
# Make jit compile
f_jitted(sq, x_test)
f_no_class_jitted(x_test)


def print_times(times, n_runs, n_repeats):
    print('{:d} loops, best of {:d}: {:.1f} usec per loop'.format(
        n_runs, n_repeats, 1e6 * min(times) / n_runs))


n_runs = 1000
n_repeats = 5

print('\nwith class:')
times = timeit.repeat(
    lambda: f(sq, x_test), repeat=n_repeats, number=n_runs)
print_times(times, n_runs, n_repeats)


print('\nwith class, jitted:')
times = timeit.repeat(
    lambda: f_jitted(sq, x_test), repeat=n_repeats, number=n_runs)
print_times(times, n_runs, n_repeats)


print('\nwithout class:')
times = timeit.repeat(
    lambda: f_no_class(x_test), repeat=n_repeats, number=n_runs)
print_times(times, n_runs, n_repeats)


print('\nwithout class, jitted:')
times = timeit.repeat(
    lambda: f_no_class_jitted(x_test), repeat=n_repeats, number=n_runs)
print_times(times, n_runs, n_repeats)


# Same math as Squarer.square, but without jax at all.
print('\noriginal numpy:')
times = timeit.repeat(
    lambda: 2 * x_test_onp**2, repeat=n_repeats, number=n_runs)
print_times(times, n_runs, n_repeats)
