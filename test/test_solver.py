import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from time import sleep

import ulfy
import matplotlib.pyplot as plt
import sympy as sp
from dolfin import *

from multirat.boundary import RobinBoundary, DirichletBoundary
from multirat.meshprocessing import Domain
from multirat.multicompartment import solve_pressure


LOGGER = logging.getLogger()


def pressure_functionspace(mesh, degree, compartments):
    P1 = FiniteElement("CG", mesh.ufl_cell(), degree)
    el = MixedElement([P1] * len(compartments))
    return FunctionSpace(mesh, el)


def assign_mixed_function(p, V, compartments):
    P = Function(V)
    subspaces = [V.sub(idx).collapse() for idx, _ in enumerate(compartments)]
    Ptrue_int = [interpolate(p[j], Vj) for j, Vj in zip(compartments, subspaces)]
    assigner = FunctionAssigner(V, subspaces)
    assigner.assign(P, Ptrue_int)
    return P


def mms_domain(N, subboundaries):
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), N, N)
    boundary_tags = MeshFunction("size_t", mesh, dim=1, value=0)
    for tag, subd in subboundaries.items():
        subd.mark(boundary_tags, tag)
    return Domain(mesh, None, boundary_tags)


def quadratic_mms(a: float, p0: float, degree: int):
    x, y = sp.symbols("x y")
    p_sym = a * (x ** 2 + y ** 2) + p0
    return p_sym


def expr(exp, degree):
    mesh_ = UnitSquareMesh(1, 1)
    V_ = FunctionSpace(mesh_, "CG", degree)
    v = Function(V_)
    return ulfy.Expression(v, subs={v: exp}, degree=degree)


def mms_parameters():
    K = defaultdict(lambda: 1.0)
    gamma = defaultdict(lambda: 1.0)
    L_bdry = defaultdict(lambda: 1.0)
    parameters = {
        "hydraulic_conductivity": K,
        "convective_fluid_transfer": gamma,
        "hydraulic_conductivity_bdry": L_bdry,
    }
    return parameters


def strong_form(p, K, gamma, degree):
    mesh_ = UnitSquareMesh(1, 1)
    V_ = FunctionSpace(mesh_, "CG", degree)
    p_ = {j: Function(V_) for j in p}
    transfer = {j: sum([gamma[(i, j)] * (p_[i] - p_[j]) for i in p if i != j]) for j in p}
    f = {j: -K[j] * div(grad(p_[j])) - transfer[j] for j in p}
    return {i: ulfy.Expression(f[i], subs={p_[j]: p[j] for j in p}, degree=degree) for i in p}


def mms_robin_boundary(pj, alpha, normals, degree):
    mesh_ = UnitSquareMesh(1, 1)
    V_ = FunctionSpace(mesh_, "CG", degree)
    p_ = Function(V_)

    gR = {tag: p_ + 1.0 / alpha * inner(grad(p_), n) for tag, n in normals.items()}
    gR = {tag: ulfy.Expression(gR[tag], subs={p_: pj}, degree=degree) for tag in gR}
    return [RobinBoundary(alpha, gR[tag], tag) for tag in gR]


def mms_dirichlet_boundary(pj, degree):
    mesh_ = UnitSquareMesh(1, 1)
    V_ = FunctionSpace(mesh_, "CG", degree)
    p_ = Function(V_)
    return [DirichletBoundary(expr(pj, degree), "everywhere")]


def expr(exp, degree):
    mesh_ = UnitSquareMesh(1, 1)
    V_ = FunctionSpace(mesh_, "CG", degree)
    v = Function(V_)
    return ulfy.Expression(v, subs={v: exp}, degree=degree)


def test_solver():
    normals = {1: Constant((-1, 0)), 2: Constant((1, 0)), 3: Constant((0, -1)), 4: Constant((0, 1))}
    subdomains = {
        1: CompiledSubDomain("near(x[0], -1)"),
        2: CompiledSubDomain("near(x[0], 1)"),
        3: CompiledSubDomain("near(x[1], -1)"),
        4: CompiledSubDomain("near(x[1], 1)"),
    }

    parameters = mms_parameters()
    K = parameters["hydraulic_conductivity"]
    gamma = parameters["convective_fluid_transfer"]
    L_bdry = parameters["hydraulic_conductivity_bdry"]

    compartments = ["pa", "pc", "pv"]
    a = {"e": 0.5, "pa": 1.0, "pc": Constant(0.0), "pv": -1.0}
    p0 = defaultdict(lambda: 0.0)
    p = {j: quadratic_mms(a[j], p0[j], degree=2) for j in compartments}
    p_expr = {i: expr(p[i], degree=2) for i in p}

    f = strong_form(p, K, gamma, degree=2)
    boundaries = {
        "pa": mms_robin_boundary(p["pa"], -L_bdry["pa"] / K["pa"], normals, degree=2),
        "pc": [],  # Homogeneous Neumann,
        "pv": mms_dirichlet_boundary(p["pv"], degree=2),
    }

    N = 10
    domain = mms_domain(N, subdomains)
    V = pressure_functionspace(domain.mesh, 2, compartments)
    Ph = solve_pressure(domain, V, compartments, boundaries, parameters, source=f)
    P = assign_mixed_function(p_expr, V, compartments)
    assert abs(P.vector()[:] - Ph.vector()[:]).max() < 1e-10


    vmin = Ph.vector().min()
    vmax = Ph.vector().max()
    fig = plt.figure()
    for idx, j in enumerate(compartments):
        fig.add_subplot(2, 2, idx + 1)
        c = plot(P.sub(idx), vmin=vmin, vmax=vmax)
        plt.colorbar(c)
        plt.title(j)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path = Path(f"/tmp/test-multirat-{timestamp}/")
    path.mkdir()
    plt.savefig(path / "mms_pressure.png")
    LOGGER.info(f"MMS pressure distributions stored at {path / 'mms_pressure.png'}")


# def test_convergence():
#     compartments, p, f, boundaries, params, subboundaries = mms_pressure("bump")
#     E = []
#     hvec = []
#     for N in [0, 1, 2, 3]:  # , 4, 5, 6, 7, 8]:
#         N = 2 ** N
#         mesh = RectangleMesh(Point(-1, -1), Point(1, 1), N, N)
#         V = pressure_functionspace(mesh, 1, compartments)
#         ph = solve_pressure(V, compartments, boundaries, params, subboundaries, source=f)
#         P = assign_mixed_function(p, V, compartments)
#         print(abs(P.vector()[:] - ph.vector()[:]).max())

#         # assert False, "above expression apparently always returns close to 0."
#         print(P.vector()[:].max())
#         print(ph.vector()[:].max())

#         E.append(
#             np.sqrt(
#                 sum([errornorm(P.sub(idx), ph.sub(idx), "H1") ** 2 for idx, j in enumerate(compartments)])
#             )
#         )
#         hvec.append(mesh.hmax())

#     vmin = ph.vector().min()
#     vmax = ph.vector().max()
#     fig = plt.figure()
#     for idx, j in enumerate(compartments):
#         # plt.figure()
#         fig.add_subplot(2, 3, idx + 1)
#         c = plot(P.sub(idx), vmin=vmin, vmax=vmax)
#         plt.colorbar(c)
#         plt.title(j)

#     # plt.figure()
#     fig.add_subplot(2, 3, 5)
#     plt.title("Error Convergence")
#     plt.loglog(hvec, E, "o-")
#     plt.xlabel("$h$")
#     plt.ylabel("$|| p - p_h||$")
#     plt.show()


if __name__ == "__main__":
    test_solver()