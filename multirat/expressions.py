from dolfin import Constant, Expression


def sqnorm_str(y: str, dim):
    if y is None:
        return f"({' + '.join([f'x[{i}] * x[{i}]' for i in range(dim)])})"
    return f"({' + '.join([f'pow(x[{i}] - {y}[{i}], 2)' for i in range(dim)])})"


def maxnorm_str(y: str, dim):
    if y is None:
        return f"max({{{', '.join([f'abs(x[{i}])' for i in range(dim)])}}})"
    return f"max({{{', '.join([f'abs(x[{i}] - {y}[{i}])' for i in range(dim)])}}})"


def characteristic_sphere(center, radius, degree=0):
    dim = len(center)
    return Expression(f"{sqnorm_str('c', dim)} <= r * r ? 1. : 0.", c=Constant(center), r=Constant(radius),
                      degree=degree)


def characteristic_cube(center, sidelength, degree=0):
    dim = len(center)
    return Expression(f"{maxnorm_str('c', dim)} <= a ? 1. : 0.", c=Constant(center), a=Constant(sidelength / 2),
                      degree=degree)
