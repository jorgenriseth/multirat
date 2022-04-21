from multirat.expressions import sqnorm_str
from dolfin import Expression, Constant


def gaussian_expression(center, std, amplitude=1.0, degree=1,  **kwargs):
    dim = len(center)
    sqnorm = f"({' + '.join([f'pow(x[{i}] - c[{i}], 2)' for i in range(dim)])})"
    sqnorm = sqnorm_str('c', dim)
    return Expression(
        f"a * exp(-{sqnorm_str('c', dim)} / (2 * b2))", a=Constant(amplitude), c=Constant(center), b2=std**2, degree=degree, **kwargs
    )
