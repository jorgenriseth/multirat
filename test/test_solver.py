import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from dolfin import *

from multirat.mms import mms_domain, mms_setup
from multirat.multicompartment import pressure_functionspace, solve_pressure
from multirat.utils import assign_mixed_function

LOGGER = logging.getLogger()


def test_solver():
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


if __name__ == "__main__":
    test_solver()
