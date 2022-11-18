from itertools import combinations
from typing import List
from pathlib import Path

from dolfin import Function, TestFunctions, Constant, VectorFunctionSpace
from dolfin import TrialFunctions, TestFunction
from dolfin import inner, grad, solve, project, assemble, lhs, rhs, dx
from pantarei.io import TimeSeriesStorage
from pantarei.boundary import process_dirichlet, process_boundary_forms
from multirat.parameters import get_base_parameters, get_interface_parameter


def compute_vectorfields(p: Function):
    V = p.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, "P", degree)
    return [project(grad(pi), W) for pi in p]


def hydraulic_conductivity(p, compartments):
    k = p["permeability"]
    mu = p["viscosity"]
    phi = p["porosity"]
    return { i: k[i] / (mu[i] * phi[i]) for i in compartments}

def convective_fluid_transfer(p , compartments):
    return get_interface_parameter(p["convective_fluid_transfer"], compartments)


def solve_pressure(domain, V, compartments, boundaries, params):
    p = TrialFunctions(V)
    q = TestFunction(V)
    K = {key: Constant(val) for key, val in hydraulic_conductivity(params, compartments).items()}
    T = {key: Constant(val) for key, val in convective_fluid_transfer(params, compartments).items()}

    imap = {label: idx for idx, label in enumerate(compartments)}
    F = 0.0
    for j in compartments:
        F += (K[j] * inner(grad(p[imap[j]]), grad(q[imap[j]]))
            - sum([ T[(i, j)]*(p[imap[i]] - p[imap[j]])*q[imap[j]] for i in compartments if i != j])
        ) * dx
    bcs = []
    for i in compartments:
        bcs.extend(process_dirichlet(domain, V.sub(imap[i]), boundaries[i]))
        F += process_boundary_forms(p[imap[i]], q[imap[i]], domain, boundaries[i])
    A = assemble(lhs(F))
    b = assemble(rhs(F))

    P = Function(V, name="pressure")
    for bc in bcs:
        bc.apply(A, b)
        solve(A, P.vector(), b)

    return P


if __name__ == "__main__":
    from dolfin import (FiniteElement, MixedElement, FunctionSpace, RectangleMesh, Point)
    from base.meshprocessing import Domain
    from multirat.parameters import get_base_parameters, get_pressure_parameters, make_dimless
    from pantarei.boundary import DirichletBoundary, RobinBoundary

    def create_mesh(n, x0=-1., y0=-1., x1=1., y1=1.):
        return Domain(
            mesh=RectangleMesh(Point(x0, y0), Point(x1, y1), n, n),
            subdomains=None,
            boundaries=None,
        )

    base = get_base_parameters()
    domain = create_mesh(100)
    compartments = ["pvs_arteries", "pvs_veins", "ecs"]

    P1 = FiniteElement('CG', domain.mesh.ufl_cell(), 1)
    el = MixedElement([P1]* len(compartments))
    V = FunctionSpace(domain.mesh, el)
    # V = LabeledFunctionSpace(domain.mesh, el, compartments)
    boundaries = {
        "pvs_arteries": [RobinBoundary(Constant(1e-1), Constant(1.0), "everywhere")],
        "pvs_veins": [DirichletBoundary(Constant(0.0), "everywhere")],
        "ecs": [DirichletBoundary(Constant(0.5), "everywhere")]
    }
    results_path = "../results/pressure"


    PARAMETER_UNITS = {
        "permeability": "mm**2",
        "viscosity": "Pa * s",
        "porosity": "",
        "convective_fluid_transfer": "1 / (Pa * s)",
        "osmotic_pressure": "Pa",
        "osmotic_reflection": ""
    }
    params = make_dimless(get_pressure_parameters(base), PARAMETER_UNITS)

    p = solve_pressure(domain, V, compartments, boundaries, params)

    storage = TimeSeriesStorage("w", "test/", mesh=domain.mesh, V=V)
    storage.write(p, 0.0)
    storage.close()

    visual = TimeSeriesStorage("r", storage.filepath)
    visual.to_pvd(compartments)
    visual.close()
