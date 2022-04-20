import skallaflow as sf
from dolfin import (Function, TestFunction, TrialFunction, Measure)
from dolfin import (assemble, dx, inner, grad, solve)

from .parameters import PARAMS
from .boundary_conditions import TracerConservationBoundary, TracerDecayBoundary, HomogeneousDirichletBoundary


class BaseDiffusionProblem:
    def __init__(self, domain, timekeeper, u0, functionspace, boundaries, D=PARAMS["diffusion_constant"]):
        self.domain = domain  # Class containing mesh, subdomains and boudnary info
        self.boundaries = boundaries
        self.time = timekeeper
        self.V = functionspace  # Function Space
        self.D = D  # Diffusion constant
        self.A = None  # Bilinear form, assembled matrix
        self.L = None  # Linear from (not assembled)
        self.u = None  # Function to hold solutions.
        self.u0 = Function(functionspace, name="concentration")  # Function to hold solution at previous timestep.
        self.u0.assign(u0)
        self.bcs = []  # List of possible DirichletBoundaries

    def build_variational_form(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        a = (u * v + self.time.dt * self.D * inner(grad(u), grad(v))) * dx
        self.A = assemble(a)
        self.L = self.u0 * v * dx

    def init_solver(self):
        self.build_variational_form()
        self.process_boundaries(self.boundaries)
        self.u = Function(self.V, name='concentration')
        self.u.assign(self.u0)

    def process_boundaries(self, boundaries):
        self.bcs = sf.process_dirichlet(self.domain, self.V, boundaries)
        self.L += sf.process_boundary_forms(self.domain, self.V, boundaries)

    def pre_solve(self):
        pass

    def solve(self):
        b = assemble(self.L)
        for bc in self.bcs:
            bc.apply(self.A, b)
        solve(self.A, self.u.vector(), b)

    def post_solve(self):
        self.u0.assign(self.u)


class HomogeneousProblem(BaseDiffusionProblem):
    def __init__(self, domain, timekeeper, u0, functionspace, D=PARAMS["diffusion_constant"]):
        bc = [HomogeneousDirichletBoundary()]
        super().__init__(domain, timekeeper, u0, functionspace, bc, D=D)


class BoundaryODEProblem(BaseDiffusionProblem):
    def pre_solve(self):
        self.boundaries[0].update(self.u0, self.time)


class TracerConservationProblem(BoundaryODEProblem):
    def __init__(self, domain, timekeeper, u0, functionspace, D=PARAMS["diffusion_constant"]):
        boundaries = [TracerConservationBoundary()]
        super().__init__(domain, timekeeper, u0, functionspace, boundaries, D)


class TracerDecayProblem(BoundaryODEProblem):
    def __init__(self, domain, timekeeper, u0, functionspace, D=PARAMS["diffusion_constant"]):
        boundaries = [TracerDecayBoundary()]
        super().__init__(domain, timekeeper, u0, functionspace, boundaries, D)
