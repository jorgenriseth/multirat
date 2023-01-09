from collections import defaultdict
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from multirat.boundary import DirichletBoundary

from multirat.computers import BaseComputer
from multirat.meshprocessing import Domain
from multirat.mms import (
    expr,
    mms_dirichlet_boundary,
    mms_domain,
    mms_parameters,
    mms_quadratic,
    mms_setup,
    mms_solute_quadratic,
    multicomp_errornorm,
    solute_sources,
    trapezoid_internal,
)
from multirat.multicompartment import pressure_functionspace, solve_pressure, solve_solute, mass_flux_density, mass_total
from multirat.parameters import to_constant
from multirat.timekeeper import TimeKeeper
from multirat.utils import assign_mixed_function

LOGGER = logging.getLogger()


def test_flux_computation():
    degree = 4
    subdomains = {
        1: CompiledSubDomain("near(x[0], -1)"),
        2: CompiledSubDomain("near(x[0], 1)"),
        3: CompiledSubDomain("near(x[1], -1)"),
        4: CompiledSubDomain("near(x[1], 1)"),
    }

    params = mms_parameters()
    phi, D, K, G, L = to_constant(
        params,
        "porosity",
        "effective_diffusion",
        "hydraulic_conductivity",
        "convective_solute_transfer",
        "diffusive_solute_transfer",
    )

    dt = 1.0
    endtime = 1.0
    compartments = ["pa", "pv"]

    ap = {"pa": -1.0, "pc": -0.5, "pv": 0.8}
    p0 = {"pa": 1.0, "pc": 0.5, "pv": 0.0}
    p_sym = {j: mms_quadratic(ap[j], p0[j]) for j in compartments}

    ac = {"pa": -1.0, "pc": -0.5, "pv": -0.2}
    c = {j: mms_solute_quadratic(ac[j], T=endtime) for j in compartments}

    time = TimeKeeper(dt, endtime)
    c_expr = {j: expr(c[j], degree=degree, t=time) for j in compartments}
    p_expr = {j: expr(p_sym[j], degree=degree) for j in compartments}
    boundaries = {j: mms_dirichlet_boundary(c[j], degree=degree, t=time) for j in compartments}

    domain = mms_domain(4, subdomains)
    V = pressure_functionspace(domain.mesh, 2, compartments)
    p = assign_mixed_function(p_expr, V, compartments)
    C0 = assign_mixed_function(c_expr, V, compartments)

    M0 = mass_total(C0, phi, compartments)

    f = solute_sources(c, p_sym, time, K, phi, D, L, G, degree)
    C = solve_solute(C0, p, time, domain, V, compartments, boundaries, params, source=f)

    sink = assemble(sum(f.values()) * Measure("dx", domain=domain.mesh))
    M1 = mass_total(C, phi, compartments)
    q = mass_flux_density(C, p, compartments, K, phi, D)
    n = FacetNormal(domain.mesh)
    Q = assemble(inner(q, n) * ds)


    print(f"Mass loss: \t {M1-M0}")
    print(f"Expected mass loss: \t {dt * (sink - Q)}")
    assert abs((M1-M0) - dt * (sink - Q)) < 1e-10


if __name__ == "__main__":
    test_flux_computation()