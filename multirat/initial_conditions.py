import skallaflow as sf
from .parameters import PARAMS
from skallaflow import DirichletInitialCondition, InitialCondition
from skallaflow import BaseProjector, DirichletProjector, AveragingDirichletProjector, NeumannProjector
from dolfin import (Expression, Constant, Function, TrialFunction,
                    TestFunction, dx, DirichletBC, solve, project,
                    Measure, assemble)


def gaussian_expression(center, std, amplitude=1.0, degree=1,  **kwargs):
    dim = len(center)
    sqnorm = f"({' + '.join([f'pow(x[{i}] - c[{i}], 2)' for i in range(dim)])})"
    return Expression(
        f"a * exp(-{sqnorm} / (2 * b2))", a=Constant(1.), c=Constant(center), b2=std**2, degree=degree, **kwargs
    )


# class GaussianInjection(InitialCondition):
#     def __init__(self, projector=None):
#         if projector is None:
#             projector = BaseProjector()
#
#         # Gaussian initial condition.
#         center = parameters.PARAMS['injection_center']
#         spread = parameters.PARAMS['injection_spread']
#         self.u0 = Expression(
#             "exp(-(pow(x[0]-s[0], 2) + pow(x[1]-s[1], 2) + pow(x[2]-s[2], 2)) / (b * b))",
#             degree=1, b=Constant(spread), s=Constant(center)
#         )
#         super().__init__(self.u0, projector)
#
#
# class HomogeneousDirichletGaussian(GaussianInjection):
#     def __init__(self):
#         super().__init__(DirichletProjector(Constant(0.)))
#
#
# class AveragingDirichletGaussian(GaussianInjection):
#     def __init__(self):
#         super().__init__(AveragingDirichletProjector())
#
#
# class NeumannGaussian(GaussianInjection):
#     def __init__(self, uN):
#         super().__init__(NeumannProjector(uN))
