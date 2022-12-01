from collections import defaultdict

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
        E.append(errornorm(P, Ph, "H1"))
        hvec.append(domain.mesh.hmax())

    vmin = Ph.vector().min()
    vmax = Ph.vector().max()

    fig = plt.figure()
    for idx, j in enumerate(compartments):
        fig.add_subplot(2, 2, idx + 1)
        c = plot(P.sub(idx), vmin=vmin, vmax=vmax)
        plt.colorbar(c)
        plt.title(j)

    plt.figure()
    plt.title("Error Convergence")
    plt.loglog(hvec, E, "o-")
    plt.xlabel("$h$")
    plt.ylabel("$||p - p_h||_{H^1}$")
    plt.show()


if __name__ == "__main__":
    run_mms_convergence()
