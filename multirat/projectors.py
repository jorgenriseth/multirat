from dolfin import (
    Constant,
    DirichletBC,
    Function,
    Measure,
    TestFunction,
    TrialFunction,
    assemble,
    dx,
    grad,
    inner,
    project,
    solve,
)


def smoothing_projection(u, V, bcs, h1_weight):
    """Projects the function u onto a function space V with 
    Dirichlet boundary conditions given by bcs with a weighted H1-norm 
    i.e.  a weight coefficient for the 1-st derivative on the norm in which
    the minimization problem is solved."""
    u_ = TrialFunction(V)
    v = TestFunction(V)
    u0_ = project(u, V)  # Projection of u onto V, without the bcs.
    a0 = u_ * v * dx + h1_weight * inner(grad(u_), grad(v)) * dx
    L0 = u0_ * v * dx + h1_weight * inner(grad(u0_), grad(v)) * dx
    u1 = Function(V)
    A = assemble(a0)
    b = assemble(L0)
    for bc in bcs:
        bc.apply(A, b)
    solve(A, u1.vector(), b)
    return u1

class BaseProjector:
    def project(self, expression, space):
        return project(expression, space)


class DirichletProjector(BaseProjector):
    """"""
    def __init__(self, uD, h1_weight):
        self.uD = uD
        self.bcs = DirichletBC()
        self.a = h1_weight

    def project(self, u0, V):
        bcs = DirichletBC(V, self.uD, "on_boundary")
        return smoothing_projection(u0, V, bcs, self.a)

class HomogeneousDirichletProjector(DirichletProjector):
    def __init__(self):
        super().__init__(Constant(0.0))


class AveragingDirichletProjector(DirichletProjector):
    def __init__(self):
        super().__init__(Constant(0.0))

    def project(self, u0, V, norm="H1"):
        ds = Measure("ds", domain=V.mesh())
        surface_area = assemble(1.0 * ds)
        self.uD = assemble(u0 * ds) / surface_area
        return super().project(u0, V, norm="H1")


def rescale_function(u: Function, value: float):
    v = u.vector()
    v *= value / assemble(u * dx)
    return u
