import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

from multirat.boundary import process_boundary_forms
from multirat.mms import (mms_domain, mms_setup, mms_quadratic, mms_solute_quadratic, solute_sources,
expr, mms_parameters, mms_dirichlet_boundary,trapezoid_internal, multicomp_errornorm,)
from multirat.multicompartment import (pressure_functionspace, solve_pressure, solute_variational_form, process_boundaries_multicompartment,
print_progress, solve_timestep)
from multirat.utils import assign_mixed_function
from multirat.parameters import to_constant


LOGGER = logging.getLogger()


def test_pressure_solver():
    p_expr, f, boundaries, parameters, subdomains, compartments = mms_setup("quadratic")

    N = 10
    domain = mms_domain(N, subdomains)
    V = pressure_functionspace(domain.mesh, 2, compartments)
    Ph = solve_pressure(domain, V, compartments, boundaries, parameters, source=f)
    P = assign_mixed_function(p_expr, V, compartments)

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
    LOGGER.info(f"MMS pressure distribution plots stored at {path / 'mms_pressure.png'}")
    assert abs(P.vector()[:] - Ph.vector()[:]).max() < 1e-10


def test_concentration_solver():
    degree = 4
    normals = {1: Constant((-1, 0)), 2: Constant((1, 0)), 3: Constant((0, -1)), 4: Constant((0, 1))}
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

    dt = 0.1
    endtime = 1.0
    compartments = ["pa", "pc", "pv"]

    ap = {"pa": -1.0, "pc": -0.5, "pv": 0.8}
    p0 = {"pa": 1.0, "pc": 0.5, "pv": 0.0}
    p_sym = {j: mms_quadratic(ap[j], p0[j]) for j in compartments}


    ac = {"pa": -1.0, "pc": -0.5, "pv": -0.2}
    c = {j: mms_solute_quadratic(ac[j], T=endtime) for j in compartments}

    time = Constant(0.0)
    f = solute_sources(c, p_sym, time, K, phi, D, L, G, degree)
    c_expr = {j: expr(c[j], degree=degree, t=time) for j in compartments}
    p_expr = {j: expr(p_sym[j], degree=degree) for j in compartments}
    boundaries = {j: mms_dirichlet_boundary(c[j], degree=degree, t=time) for j in compartments}

    domain = mms_domain(4, subdomains)
    V = pressure_functionspace(domain.mesh, 2, compartments)
    p = assign_mixed_function(p_expr, V, compartments)
    C0 = assign_mixed_function(c_expr, V, compartments)

    c = TrialFunctions(V)
    w = TestFunction(V)
    phi, D, K, G, L = to_constant(
        params,
        "porosity",
        "effective_diffusion",
        "hydraulic_conductivity",
        "convective_solute_transfer",
        "diffusive_solute_transfer",
    )
    F_bdry, bcs = process_boundaries_multicompartment(c, w, boundaries, V, compartments, domain)
    F = solute_variational_form(c, w, compartments, dt, C0, p, D, K, L, G, phi, source=f) - dt * F_bdry
    
    a = lhs(F)
    l = rhs(F)
    A = assemble(a)

    Ch = Function(V, name="concentration")
    errs = np.nan * np.zeros(100)
    idx = 0
    errs[idx] = multicomp_errornorm(c_expr, C0, compartments, "H1")
    while float(time) <= endtime:
        time.assign(time + dt)
        print_progress(float(time), endtime)
        Ch = solve_timestep(A, l, Ch, bcs)
        C0.assign(Ch)

        C0_ = assign_mixed_function(c_expr, V, compartments)
        err = abs(Ch.vector()[:] - C0_.vector()[:]).max()
        print(err)
        idx+=1
        errs[idx] = multicomp_errornorm(c_expr, C0, compartments, "H1")

    errs = errs[~np.isnan(errs)]    
    temporal_error = np.sqrt(trapezoid_internal(errs**2, dt))
    assert temporal_error < 1e-10


if __name__ == "__main__":
    test_pressure_solver()
    test_concentration_solver()
