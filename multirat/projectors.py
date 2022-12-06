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


class BaseProjector:
    def project(self, expression, space):
        return project(expression, space)


class DirichletProjector(BaseProjector):
    def __init__(self, uD):
        self.uD = uD
        self.bcs = DirichletBC()

    def project(self, u0, V, norm="H1"):
        bcs = DirichletBC(V, self.uD, "on_boundary")
        if norm == "H1":
            # Project u0 to have Dirichlet boundary equal to g0.
            u = TrialFunction(V)
            v = TestFunction(V)
            u0_ = project(u0, V)
            a0 = u * v * dx + inner(grad(u), grad(v)) * dx
            L0 = u0_ * v * dx + inner(grad(u0_), grad(v)) * dx
            u1 = Function(V)
            A = assemble(a0)
            b = assemble(L0)
            bcs.apply(A, b)
            solve(A, u1.vector(), b)
            return u1
        elif norm == "L2":
            return project(u0, V, bcs=bcs)
        raise ValueError(f"norm should be 'H1' or 'L2' got '{norm}'.")


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
