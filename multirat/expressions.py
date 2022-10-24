""" This file contains various helper functions for construction of expressions and characteristic functions. """
from dolfin import Constant, Expression


def sqnorm_cpp_code(y: str, dim):
    """Create a C++ compatible expression for computing the squared euclidean
    distance between a point x in dimension 'dim', and the input string y. (eg. ||x - 'center'||).
    if y is None, it computes the squared norm of x."""
    if y is None:
        return f"({' + '.join([f'x[{i}] * x[{i}]' for i in range(dim)])})"
    return f"({' + '.join([f'pow(x[{i}] - {y}[{i}], 2)' for i in range(dim)])})"


def maxnorm_cpp_code(y: str, dim):
    if y is None:
        return f"max({{{', '.join([f'abs(x[{i}])' for i in range(dim)])}}})"
    return f"max({{{', '.join([f'abs(x[{i}] - {y}[{i}])' for i in range(dim)])}}})"


def characteristic_sphere(center, radius, degree=0):
    """Create an expression for the characteristic function of a ball with given center and radius."""
    dim = len(center)
    return Expression(f"{sqnorm_cpp_code('c', dim)} <= r * r ? 1. : 0.", c=Constant(center), r=Constant(radius),
                      degree=degree)


def characteristic_cube(center, sidelength, degree=0):
    """Create an expression for the characteristic function of a cube with given center and sidelength."""
    dim = len(center)
    return Expression(f"{maxnorm_cpp_code('c', dim)} <= a ? 1. : 0.", c=Constant(center), a=Constant(sidelength / 2),
                      degree=degree)
