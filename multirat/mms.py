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
    errornorm,
    grad,
    inner,
    interpolate,
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


def expr(exp, degree, **kwargs):
    v = mms_placeholder()
    return ulfy.Expression(v, subs={v: exp}, degree=degree, **kwargs)


def mms_sources(p, K, gamma, degree):
    p_ = {j: mms_placeholder() for j in p}
    transfer = {j: sum([gamma[(i, j)] * (p_[i] - p_[j]) for i in p if i != j]) for j in p}
    f = {j: -K[j] * div(grad(p_[j])) - transfer[j] for j in p}
    return {i: ulfy.Expression(f[i], subs={p_[j]: p[j] for j in p}, degree=degree) for i in p}


def mms_parameters():
    return defaultdict(lambda: defaultdict(lambda: 1.0))


def mms_placeholder():
    mesh_ = UnitSquareMesh(1, 1)
    V_ = FunctionSpace(mesh_, "CG", 1)
    return Function(V_)


def mms_robin_boundary(p, alpha, normals, degree):
    p_ = mms_placeholder()
    gR = {tag: p_ + 1.0 / alpha * inner(grad(p_), n) for tag, n in normals.items()}
    gR = {tag: ulfy.Expression(gR[tag], subs={p_: p}, degree=degree) for tag in gR}
    return [RobinBoundary(alpha, gR[tag], tag) for tag in gR]


def mms_dirichlet_boundary(p, degree, **kwargs):
    return [DirichletBoundary(expr(p, degree, **kwargs), "everywhere")]


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


def solute_transfer(c_, p_, L, G, phi):
    s = {}
    for j in c_:
        s[j] = 0.0
        for i in c_:
            if i == j:
                continue
            s[j] += L[(i, j)] * (c_[i] - c_[j]) + 0.5 * G[(i, j)] * (c_[i] + c_[j]) * (p_[i] - p_[j])
    return s


def solute_sources(c, p, time, K, phi, D, L, G, degree):
    c_ = {j: mms_placeholder() for j in c}
    dcdt_ = {j: mms_placeholder() for j in c}
    p_ = {j: mms_placeholder() for j in c}
    s = solute_transfer(c_, p_, L, G, phi)
    f = {
        j: dcdt_[j] - (K[j] / phi[j] * div(c_[j] * grad(p_[j]))) - (D[j] * div(grad(c_[j]))) - s[j] / phi[j]
        for j in c
    }
    subs = {
        **{dcdt_[j]: sp.diff(c[j], "t") for j in c},
        **{c_[j]: c[j] for j in c},
        **{p_[j]: p[j] for j in c},
    }
    return {i: ulfy.Expression(f[i], subs=subs, degree=degree, t=time) for i in c}


def mms_solute_quadratic(a, T):
    c0 = a / 2.0
    t, x, y = sp.symbols("t x y")
    c_sym = a * (1.0 - t / T) * (x ** 2 + y ** 2) + c0
    return c_sym


def trapezoid_internal(f, h):
    return h * (0.5 * (f[0] + f[-1]) + f[1:-1].sum())


def multicomp_errornorm(u, uh, compartments, norm="H1"):
    Vhigh = FunctionSpace(uh.function_space().mesh(), "CG", 5)
    return sum([errornorm(interpolate(u[j], Vhigh), uh.sub(idx), norm) for idx, j in enumerate(compartments)])
