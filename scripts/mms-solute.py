from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dolfin import CompiledSubDomain

from multirat.parameters import to_constant
from multirat.multicompartment import pressure_functionspace, solve_solute

from multirat.utils import assign_mixed_function
from multirat.mms import *
from multirat.computers import BaseComputer
from multirat.timekeeper import TimeKeeper


class ErrorComputer(BaseComputer):
    def __init__(self, c_expr, compartments):
        super().__init__({"errornorm": lambda c: multicomp_errornorm(c_expr, c, compartments, "H1")})


def run_mms_solute_convergence():
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
    time = TimeKeeper(dt=0.1, endtime=1.0)
    compartments = ["pa", "pc", "pv"]

    # Pressures in sympy format.
    ap = {"pa": -1.0, "pc": -0.5, "pv": 0.8}
    p0 = {"pa": 1.0, "pc": 0.5, "pv": 0.0}
    p_sym = {j: mms_quadratic(ap[j], p0[j]) for j in compartments}

    # Concentrations in sympy format.
    ac = {"pa": -1.0, "pc": -0.5, "pv": -0.2}
    c = {j: mms_solute_quadratic(ac[j], T=time.endtime) for j in compartments}

    # Define solute sources and expressions for sympy represnetations.
    f = solute_sources(c, p_sym, time, K, phi, D, L, G, degree)
    c_expr = {j: expr(c[j], degree=degree, t=time) for j in compartments}
    p_expr = {j: expr(p_sym[j], degree=degree) for j in compartments}

    # Define boundary conditions
    boundaries = {j: mms_dirichlet_boundary(c[j], degree=degree, t=time) for j in compartments}

    E = []
    hvec = []
    for N in [3, 4, 5, 6]:
        # Setup problem
        time.reset()
        domain = mms_domain(2 ** N, subdomains)
        V = pressure_functionspace(domain.mesh, 1, compartments)
        p = assign_mixed_function(p_expr, V, compartments)
        C0 = assign_mixed_function(c_expr, V, compartments)
        computer = ErrorComputer(c_expr, compartments)
        # Solve Problem
        solve_solute(C0, p, time, domain, V, compartments, boundaries, params, source=f, computer=computer)
        # Compute error
        temporal_error = np.sqrt(trapezoid_internal(computer["errornorm"] ** 2, time.dt))
        E.append(temporal_error)
        hvec.append(domain.mesh.hmax())
    hvec, E = np.array(hvec), np.array(E)

    savepath = (Path() / "../results/mms").resolve()
    savepath.mkdir(exist_ok=True)

    plt.figure()
    plt.loglog(hvec, E, "-o", markeredgecolor="k")
    xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.autoscale(False)
    plt.loglog(hvec, hvec, "k--", lw=0.5, label="Linear")
    plt.ylim(ylim)

    plt.xlabel("$h$")
    plt.ylabel(r"$||c - c||_{L^2([0, T], H^1(\Omega)}$")
    plt.legend()
    plt.savefig(savepath / "convergence-concentration.png")
    plt.show()


if __name__ == "__main__":
    run_mms_solute_convergence()
