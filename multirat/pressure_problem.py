from typing import List, Dict, Any

from dolfin import Function, TestFunction, TrialFunction, FunctionSpace
from dolfin import assemble, dx, inner, grad, solve

from multirat.base.boundary import (
    BoundaryData,
    process_dirichlet,
    process_boundary_forms,
)

from multirat.base.meshprocessing import Domain
from multirat.base.timekeeper import TimeKeeper
from pantarei.io import BaseComputer
from multirat.diffusion import BaseDiffusionProblem


# def solve_stationary_problem(problem: BaseDiffusionProblem, results_path, computer=None):
#     if computer is None:
#         computer = BaseComputer({})

#     problem.init_solver()
#     computer.initiate(problem)
#     storage = TimeSeriesStorage("w", results_path, mesh=problem.domain.mesh, V=problem.V)
    
#     problem.solve()
#     storage.write(problem.u)
#     computer.compute(problem)
#     storage.close()
#     print()


class MultiCompartmentPressureProblem:
    def __init__(self, domain: Domain, functionspace: FunctionSpace,  compartments: List[str], u0: Function, boundaries: List[BoundaryData], parameters: Dict[Any, Any]):
        self.domain = domain  # Class containing mesh, subdomains and boudnary info
        self.boundaries = boundaries
        self.time = timekeeper
        self.V = functionspace  # Function Space
        self.A = None  # Bilinear form, assembled matrix
        self.L = None  # Linear from (not assembled)
        self.u = None  # Function to hold solutions.
        self.u0 = Function(functionspace, name="concentration")  # Function to hold solution at previous timestep.
        self.u0.assign(u0)
        self.bcs = []  # List of possible DirichletBoundaries


    def build_variational_form(self, parameters: Dict):
        if not V.num_sub_spaces() == len(self.compartments):
            raise ValueError("V should be MixedElement space of same dimension as the number of compartments.")
        q = TestFunctions(p.function_space())
        k = {key: Constant(val) for key, val in compute_darcy_constants(parameters, compartments).items()}
        T = {key: Constant(val) for key, val in get_transfer_coefficients("transfer_fluid", parameters, compartments).items()}
        a = 0.
        for j, comp_j in enumerate(compartments):
            a += (
                k[comp_j] * inner(grad(p[j]), grad(q[j])) 
                - sum([ T[(comp_i, comp_j)]*(p[i] - p[j])*q[j] for i, comp_i in enumerate(compartments) if i != j])
            ) * dx
        return a, None

    def init_solver(self):
        self.build_variational_form()
        self.process_boundaries(self.boundaries)
        self.P = Function(self.V, name="pressure")

    def process_boundaries(self, boundaries):
        self.bcs = process_dirichlet(self.domain, self.V, boundaries)
        self.L += process_boundary_forms(self.domain, self.V, boundaries)

    def solve(self):
        b = assemble(self.L)
        for bc in self.bcs:
            bc.apply(self.A, b)
        solve(self.A, self.u.vector(), b)