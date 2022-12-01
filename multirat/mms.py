from collections import defaultdict

import sympy as sp
import ulfy
from dolfin import (
    CompiledSubDomain,
    Constant,
    Function,
    FunctionSpace,
    MeshFunction,
    Point,
    RectangleMesh,
    UnitSquareMesh,
    div,
    grad,
    inner,
)

from multirat.boundary import DirichletBoundary, RobinBoundary
from multirat.meshprocessing import Domain
from multirat.utils import assign_mixed_function


def mms_domain(N, subboundaries):
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), N, N)
    boundary_tags = MeshFunction("size_t", mesh, dim=1, value=0)
    for tag, subd in subboundaries.items():
        subd.mark(boundary_tags, tag)
    return Domain(mesh, None, boundary_tags)


def mms_quadratic(a: float, p0: float):
    x, y = sp.symbols("x y")
    p_sym = a * (x ** 2 + y ** 2) + p0
    return p_sym


def mms_bump(a: float, p0: float):
    x, y = sp.symbols("x y")
    p_sym = a * sp.cos(sp.pi * x / 2.0) * sp.cos(sp.pi * y / 2.0) + p0
    return p_sym


def expr(exp, degree):
    v = mms_placeholder()
    return ulfy.Expression(v, subs={v: exp}, degree=degree)


def mms_sources(p, K, gamma, degree):
    p_ = {j: mms_placeholder() for j in p}
    transfer = {j: sum([gamma[(i, j)] * (p_[i] - p_[j]) for i in p if i != j]) for j in p}
    f = {j: -K[j] * div(grad(p_[j])) - transfer[j] for j in p}
    return {i: ulfy.Expression(f[i], subs={p_[j]: p[j] for j in p}, degree=degree) for i in p}


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


def mms_placeholder():
    mesh_ = UnitSquareMesh(1, 1)
    V_ = FunctionSpace(mesh_, "CG", 1)
    return Function(V_)


def mms_robin_boundary(pj, alpha, normals, degree):
    p_ = mms_placeholder()
    gR = {tag: p_ + 1.0 / alpha * inner(grad(p_), n) for tag, n in normals.items()}
    gR = {tag: ulfy.Expression(gR[tag], subs={p_: pj}, degree=degree) for tag in gR}
    return [RobinBoundary(alpha, gR[tag], tag) for tag in gR]


def mms_dirichlet_boundary(pj, degree):
    return [DirichletBoundary(expr(pj, degree), "everywhere")]


def mms_setup(func: str, degree=4):
    normals = {1: Constant((-1, 0)), 2: Constant((1, 0)), 3: Constant((0, -1)), 4: Constant((0, 1))}
    subdomains = {
        1: CompiledSubDomain("near(x[0], -1)"),
        2: CompiledSubDomain("near(x[0], 1)"),
        3: CompiledSubDomain("near(x[1], -1)"),
        4: CompiledSubDomain("near(x[1], 1)"),
    }
    compartments = ["e", "pa", "pc", "pv"]
    a = {"e": -0.25, "pa": -1.0, "pc": 0.0, "pv": -1.0}
    p0 = {"e": 0.6, "pa": 1.0, "pc": 0.55, "pv": 0.0}
    if func == "quadratic":
        p = {j: mms_quadratic(a[j], p0[j]) for j in compartments}
    elif func == "bump":
        p = {j: mms_bump(a[j], p0[j]) for j in compartments}
    else:
        ValueError("func should be 'quadratic' or 'bump', got '{}'".format(func))

    parameters = mms_parameters()
    K = parameters["hydraulic_conductivity"]
    gamma = parameters["convective_fluid_transfer"]
    L_bdry = parameters["hydraulic_conductivity_bdry"]
    p_expr = {i: expr(p[i], degree=degree) for i in p}
    f = mms_sources(p, K, gamma, degree=degree)
    boundaries = {
        "e": mms_robin_boundary(p["e"], -L_bdry["pa"] / K["pa"], normals, degree=degree),
        "pa": mms_robin_boundary(p["pa"], -L_bdry["pa"] / K["pa"], normals, degree=degree),
        "pc": [],  # Homogeneous Neumann,
        "pv": mms_dirichlet_boundary(p["pv"], degree=degree),
    }
    return p_expr, f, boundaries, parameters, subdomains, compartments
