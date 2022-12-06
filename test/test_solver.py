import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

from multirat.computers import BaseComputer
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
from multirat.multicompartment import pressure_functionspace, solve_pressure, solve_solute
from multirat.parameters import to_constant
from multirat.timekeeper import TimeKeeper
from multirat.utils import assign_mixed_function

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


class ErrorComputer(BaseComputer):
    def __init__(self, c_expr, compartments):
        super().__init__({"errornorm": lambda c: multicomp_errornorm(c_expr, c, compartments, "H1")})


def test_concentration_solver():
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

    dt = 0.1
    endtime = 1.0
    compartments = ["pa", "pc", "pv"]

    ap = {"pa": -1.0, "pc": -0.5, "pv": 0.8}
    p0 = {"pa": 1.0, "pc": 0.5, "pv": 0.0}
    p_sym = {j: mms_quadratic(ap[j], p0[j]) for j in compartments}

    ac = {"pa": -1.0, "pc": -0.5, "pv": -0.2}
    c = {j: mms_solute_quadratic(ac[j], T=endtime) for j in compartments}

    time = TimeKeeper(dt, endtime)
    f = solute_sources(c, p_sym, time, K, phi, D, L, G, degree)
    c_expr = {j: expr(c[j], degree=degree, t=time) for j in compartments}
    p_expr = {j: expr(p_sym[j], degree=degree) for j in compartments}
    boundaries = {j: mms_dirichlet_boundary(c[j], degree=degree, t=time) for j in compartments}

    domain = mms_domain(4, subdomains)
    V = pressure_functionspace(domain.mesh, 2, compartments)
    p = assign_mixed_function(p_expr, V, compartments)
    C0 = assign_mixed_function(c_expr, V, compartments)

    computer = ErrorComputer(c_expr, compartments)
    solve_solute(C0, p, time, domain, V, compartments, boundaries, params, source=f, computer=computer)
    temporal_error = np.sqrt(trapezoid_internal(computer["errornorm"] ** 2, dt))
    assert temporal_error < 1e-10


if __name__ == "__main__":
    test_pressure_solver()
    test_concentration_solver()
