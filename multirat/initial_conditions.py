from dolfin import Constant, Expression

from multirat.expressions import cpp_sqnorm


def gaussian_expression(center, std, amplitude=1.0, degree=1, **kwargs):
    dim = len(center)
    if "domain" in kwargs:
        assert dim == kwargs["domain"].geometry().dim()
    return Expression(
        f"a * exp(-{cpp_sqnorm('c', dim)} / (2 * b2))",
        a=Constant(amplitude),
        c=Constant(center),
        b2=std ** 2,
        degree=degree,
        **kwargs,
    )
