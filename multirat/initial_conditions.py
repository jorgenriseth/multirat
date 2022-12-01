from dolfin import Constant, Expression

from multirat.expressions import sqnorm_cpp_code


def gaussian_expression(center, std, amplitude=1.0, degree=1, **kwargs):
    dim = len(center)
    return Expression(
        f"a * exp(-{sqnorm_cpp_code('c', dim)} / (2 * b2))",
        a=Constant(amplitude),
        c=Constant(center),
        b2=std ** 2,
        degree=degree,
        **kwargs,
    )
