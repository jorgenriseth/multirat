from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ulfy
from dolfin import *

from multirat.mms import mms_domain, mms_setup
from multirat.multicompartment import pressure_functionspace, solve_pressure
from multirat.utils import assign_mixed_function


def run_mms_convergence():
    p_expr, f, boundaries, parameters, subdomains, compartments = mms_setup("bump")
    E = []
    hvec = []
    for N in [1, 2, 3, 4, 5, 6, 7, 8]:
        N = 2 ** N
        domain = mms_domain(N, subdomains)
        V = pressure_functionspace(domain.mesh, 1, compartments)
        Ph = solve_pressure(domain, V, compartments, boundaries, parameters, source=f)
        P = assign_mixed_function(p_expr, V, compartments)
        E.append(errornorm(P, Ph, "H1"))  # TODO: Replace by custom function, NOT interpolating onto V
        hvec.append(domain.mesh.hmax())

    vmin = Ph.vector().min()
    vmax = Ph.vector().max()

    savepath = (Path(__file__).parent / "../results/mms").resolve()
    savepath.mkdir(exist_ok=True)

    fig = plt.figure()
    for idx, j in enumerate(compartments):
        fig.add_subplot(2, 2, idx + 1)
        c = plot(Ph.sub(idx), vmin=vmin, vmax=vmax)
        plt.colorbar(c)
        plt.title(j)
    plt.savefig(savepath / "pressure-distr.png")

    hvec = np.array(hvec)
    plt.figure()
    # plt.title("Error Convergence")
    plt.loglog(hvec, E, "o-", markeredgecolor='k')
    plt.xlabel("$h$")
    plt.ylabel("$||p - p_h||_{H^1}$")

    xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.autoscale(False)
    plt.loglog([1e-2, 1e0], [1e-3, 1e1], 'k--', lw=0.5, label="Quadratic")
    # plt.loglog([1e-2, 1e0], [1e-4, 1e-2], 'k--', lw=0.5, label="Linear")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.savefig(savepath / "convergence.png")
    plt.show()

    return hvec, E


if __name__ == "__main__":
    run_mms_convergence()
