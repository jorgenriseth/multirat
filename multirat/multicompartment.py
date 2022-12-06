from pathlib import Path

import numpy as np
from dolfin import *

from multirat.boundary import process_boundary_forms, process_dirichlet
from multirat.meshprocessing import Domain
from multirat.parameters import to_constant
from multirat.timeseriesstorage import TimeSeriesStorage


def pressure_variational_form(trial, test, compartments, K, G, source=None):
    """
    source: dictionary of source termss labeled by compartment.
    """
    p = trial
    q = test
    F = 0.0
    for idx_j, j in enumerate(compartments):
        F += K[j] * inner(grad(p[idx_j]), grad(q[idx_j])) * dx
        F -= (
            sum(
                [
                    G[(i, j)] * (p[idx_i] - p[idx_j]) * q[idx_j]
                    for idx_i, i in enumerate(compartments)
                    if idx_i != idx_j
                ]
            )
            * dx
        )
    if source is not None:
        F -= sum([source[j] * q[idx] for idx, j in enumerate(compartments)]) * dx
    return F


def solve_stationary(V, F, bcs, name="pressure"):
    A = assemble(lhs(F))
    b = assemble(rhs(F))
    P = Function(V, name=name)
    for bc in bcs:
        bc.apply(A, b)
    solve(A, P.vector(), b)
    return P

def solve_timestep(A, l: Form, u: Function, bcs):
    b = assemble(l)
    for bc in bcs:
        bc.apply(A, b)
    solve(A, u.vector(), b)
    return u


def process_boundaries_multicompartment(p, q, boundaries, V, compartments, domain):
    bcs = []
    F_bdry = 0.0
    for idx_i, i in enumerate(compartments):
        bcs.extend(process_dirichlet(domain, V.sub(idx_i), boundaries[i]))
        F_bdry -= process_boundary_forms(p[idx_i], q[idx_i], domain, boundaries[i])
    return F_bdry, bcs


def solve_pressure(domain, V, compartments, boundaries, params, source=None):
    p = TrialFunctions(V)
    q = TestFunction(V)
    K, G = to_constant(params, "hydraulic_conductivity", "convective_fluid_transfer")

    F_bdry, bcs = process_boundaries_multicompartment(p, q, boundaries, V, compartments, domain)
    F = pressure_variational_form(p, q, compartments, K, G, source=source) + F_bdry
    return solve_stationary(V, F, bcs, "pressure")


def pressure_functionspace(mesh, degree, compartments):
    P1 = FiniteElement("CG", mesh.ufl_cell(), degree)
    el = MixedElement([P1] * len(compartments))
    return FunctionSpace(mesh, el)


def store_pressure(filepath, p, compartments, store_xdmf=False):
    V = p.function_space()
    mesh = V.mesh()
    storage = TimeSeriesStorage("w", str(filepath), mesh=mesh, V=V)
    storage.write(p, 0.0)
    storage.close()
    if store_xdmf:
        visual = TimeSeriesStorage("r", storage.filepath)
        visual.to_xdmf(compartments)
        visual.close()


def total_mass(c, phi, compartments):
    return sum([assemble(phi[i] * c[idx] * dx) for idx, i in enumerate(compartments)])


def print_progress(t, T):
    progress = int(20 * t / T)
    print(f"[{'=' * progress}{' ' * (20 - progress)}] {t / 60:>6.1f}min / {T / 60:<5.1f}min", end="\r")


def solute_variational_form(trial, test, compartments, dt, c0, p, D, K, L, G, phi, source=None):
    c = trial
    w = test
    F = 0.0
    for idj, j in enumerate(compartments):
        F += (c[idj] - c0[idj]) * w[idj] * dx
        F += dt * (inner(D[j] * grad(c[idj]) + K[j] / phi[j] * c[idj] * grad(p[idj]), grad(w[idj]))) * dx
        sj = 0.0
        for idi, i in enumerate(compartments):
            if idi == idj:
                continue
            sj += (
                L[(i, j)] * (c[idi] - c[idj])
                + G[(i, j)] * (p[idi] - p[idj]) * (0.5 * (c[idi] + c[idj]))
            ) / phi[j]
        F -= dt * sj * w[idj] * dx
        if source is not None:
            F -= dt * source[j] * w[idj] * dx
    return F

def update_time(expr, newtime):
    if hasattr(expr, "t"):
        expr.t.assign(newtime)


def solve_solute(
    c0,
    p,
    dt,
    T,
    domain,
    V,
    compartments,
    boundaries,
    params,
    results_path="../results/concentration/",
    source=None,
):
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

    C0 = project(c0, V)
    F = solute_variational_form(c, w, compartments, p, C0, phi, D, K, G, L, source=source)

    F, bcs = process_boundaries_multicompartment(c, w, F, boundaries, V, compartments, domain)
    a = lhs(F)
    l = rhs(F)
    A = assemble(a)

    storage = TimeSeriesStorage("w", results_path, mesh=domain.mesh, V=V)
    t = Constant(0.0)
    storage.write(C0, t)

    idx = 0
    mass = np.nan * np.zeros(int(T / dt) + 1)
    mass[idx] = total_mass(C0, phi, compartments)

    C = Function(V, name="concentration")
    while float(t) < T:
        print_progress(float(t), T)
        C = solve_timestep(A, l, C, bcs)
        C0.assign(C)
        t.assign(t + dt)
        storage.write(C, t)
        idx += 1
        mass[idx] = total_mass(C0, phi, compartments)
    storage.close()

    visual = TimeSeriesStorage("r", storage.filepath)
    visual.to_xdmf(compartments)
    visual.close()

    np.savetxt(Path(results_path).resolve() / "mass.txt", mass)
    return C, mass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from multirat.boundary import DirichletBoundary, RobinBoundary
    from multirat.meshprocessing import Domain
    from multirat.parameters import multicompartment_parameters

    def create_mesh(n, x0=-1.0, y0=-1.0, x1=1.0, y1=1.0):
        return Domain(
            mesh=RectangleMesh(Point(x0, y0), Point(x1, y1), n, n), subdomains=None, boundaries=None
        )

    compartments = ["ecs", "pvs_arteries", "pvs_capillaries", "pvs_veins"]
    params = multicompartment_parameters(compartments)

    domain = create_mesh(40)
    V = pressure_functionspace(domain.mesh, degree=1, compartments=compartments)

    # Boundary conditions
    L_bdry = params["hydraulic_conductivity_bdry"]
    p_bdry = params["pressure_boundaries"]
    boundaries = {
        "ecs": [RobinBoundary(L_bdry["ecs"], p_bdry["ecs"], "everywhere")],
        "pvs_arteries": [RobinBoundary(L_bdry["pvs_arteries"], p_bdry["pvs_arteries"], "everywhere")],
        "pvs_capillaries": [],
        "pvs_veins": [DirichletBoundary(p_bdry["pvs_veins"], "everywhere")],
    }

    results_path = Path("../results/pressure").resolve()
    p = solve_pressure(domain, V, compartments, boundaries, params)
    store_pressure(results_path, p, compartments, store_xdmf=True)

    solute_bcs = {
        "ecs": [DirichletBoundary(Constant(0.0), "everywhere")],
        "pvs_arteries": [DirichletBoundary(Constant(0.0), "everywhere")],
        "pvs_capillaries": [DirichletBoundary(Constant(0.0), "everywhere")],
        "pvs_veins": [DirichletBoundary(Constant(0.0), "everywhere")],
    }

    dt = 600  # 10min timestep.
    T = 1 * 3600  # 4h
    N = int(T / dt) + 1
    c_init = Expression(
        "exp(-(pow(x[0] - c[0], 2) + pow(x[1]-c[1], 2)) / (length * length))",
        length=Constant(0.5),
        c=Constant((0.5, 0.0)),
        degree=2,
    )

    c0 = Function(V)
    assign(c0.sub(0), project(c_init, V.sub(0).collapse()))

    times = np.array([dt * i for i in range(N)])
    C, mass = solve_solute(c0, p, dt, T, domain, V, compartments, solute_bcs, params)

    plt.figure()
    plt.plot(times, mass, "o-")
    plt.show()
