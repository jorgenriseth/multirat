from collections import defaultdict


import ulfy
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *


from multirat.multicompartment import pressure_functionspace
from multirat.boundary import RobinBoundary, DirichletBoundary


def mms_domain(N, subboundaries):
    mesh = UnitSquareMesh(N, N)
    boundaries = MeshFunction("size_t", mesh, 1, 0)
    [subd.mark(boundaries, tag) for tag, subd in subboundaries.items()]
    return boundaries


def mms_pressure(compartments, solution="quadratic"):
    # Define unit normal vector of boundary (unit square).
    normals = {1: Constant((-1, 0)), 2: Constant((1, 0)), 3: Constant((0, -1)), 4: Constant((0, 1))}

    # Define various subdomains.
    subdomains = {
        1: CompiledSubDomain("near(x[0], -1)"),
        2: CompiledSubDomain("near(x[0], 1)"),
        3: CompiledSubDomain("near(x[1], -1)"),
        4: CompiledSubDomain("near(x[1], 1)"),
    }

    # Define MMS Place holders
    x, y = SpatialCoordinate(UnitSquareMesh(1, 1))  # Placeholder coordinates

    # Compartments
    compartments = ["e", "pa", "pv", "pc"]

    # Prepare coefficients
    K = defaultdict(lambda: 1.0)
    gamma = defaultdict(lambda: 1.0)
    L_bdry = defaultdict(lambda: 1.0)
    p0 = defaultdict(lambda: 0.0)

    # Define solutions
    a = {"e": 0.5, "pa": 1.0, "pc": Constant(0.0), "pv": -1.0}
    if solution == "quadratic":
        p = {j: a[j] * cos(pi * x / 2) * cos(pi * y / 2) + p0[j] for j in compartments}
    elif solution == "bump":
        p = {j: a[j] * (x ** 2 + y ** 2) + p0[j] for j in compartments}
    else:
        raise ValueError("solution should be 'quadratic' or 'bump', got '{}'".format(solution))

    # Define remaining terms
    transfer = {j: sum([gamma[(i, j)] * (p[i] - p[j]) for i in compartments if i != j]) for j in compartments}
    f = {j: -K[j] * div(grad(p[j])) - transfer[j] for j in compartments}
    gR = {
        j: {tag: L_bdry[j] * p[j] + K[j] * inner(grad(p[j]), n) for tag, n in normals.items()}
        for j in compartments
    }
    gN = {j: {tag: -K[j] * inner(grad(p[j]), n) for tag, n in normals.items()} for j in compartments}
    gD = {j: {tag: p[j] for tag, _ in normals.items()} for j in compartments}

    expr = lambda v: ulfy.Expression(v, degree=4)  # Here we set it
    p_true, f_true = map(lambda v: {j: expr(v[j]) for j in compartments}, (p, f))
    for j in compartments:
        gR[j] = {tag: expr(gR[j][tag]) for tag in subdomains}
        gN[j] = {tag: expr(gN[j][tag]) for tag in subdomains}
        gD[j] = {tag: expr(gD[j][tag]) for tag in subdomains}

    boundaries = {
        "e": [RobinBoundary(L_bdry["e"], gR["e"][tag], tag) for tag in subdomains],
        "pa": [RobinBoundary(L_bdry["pa"], gR["pa"][tag], tag) for tag in subdomains],
        "pc": [],
        "pv": [DirichletBoundary(gD["pv"][tag], tag) for tag in subdomains],
    }

    parameters = {
        "hydraulic_conductivity": K,
        "convective_fluid_transfer": gamma,
        "hydraulic_conductivity_bdry": L_bdry,
    }

    return compartments, p_true, f_true, boundaries, parameters, subdomains


def assign_mixed_function(p_true, V, compartments):
    P = Function(V)
    subspaces = [V.sub(idx).collapse() for idx, _ in enumerate(compartments)]
    Ptrue_int = [interpolate(p_true[j], Vj) for j, Vj in zip(compartments, subspaces)]
    assigner = FunctionAssigner(V, subspaces)
    assigner.assign(P, Ptrue_int)
    return P


from multirat.multicompartment import solve_pressure
from multirat.meshprocessing import Domain


def test_solver():
    # Use quadratic MMS
    compartments, p, f, boundaries, params, subboundaries = mms_pressure("quadratic")

    N = 2 ** 6
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), N, N)
    V = pressure_functionspace(mesh, 2, compartments)
    ph = solve_pressure(V, compartments, boundaries, params, subboundaries, source=f)
    P = assign_mixed_function(p, V, compartments)

    assert (P.vector()[:] - ph.vector()[:]).max() < 1e-10


if __name__ == "__main__":
    test_solver()
