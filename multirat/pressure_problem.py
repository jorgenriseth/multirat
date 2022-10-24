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


class MultiCompartmentPressureProblem:
    def __init__(
        self,
        domain: Domain,
        timekeeper: TimeKeeper,
        u0: Function,
        functionspace: FunctionSpace,
        boundaries: List[BoundaryData],
        parameters: Dict[Any, Any],
    ):
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

def single_compartment_variatonal_form(V: FunctionSpace, compartments: Dict[str, float]):
    u = TrialFunction(V)
    v = TrialFunction(V)