from pathlib import Path

import numpy as np

from dolfin import *
from multirat.boundary import process_dirichlet, process_boundary_forms
from multirat.parameters import to_constant
from multirat.timeseriesstorage import TimeSeriesStorage
from multirat.meshprocessing import Domain


def pressure_variational_form(trial, test, compartments, K, G, source=None):
    """
    source: dictionary of source termszs labeled by compartment.
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


def process_boundaries_multicompartment(p, q, F, boundaries, V, compartments, domain):
    bcs = []
    for idx_i, i in enumerate(compartments):
        bcs.extend(process_dirichlet(domain, V.sub(idx_i), boundaries[i]))
        F -= process_boundary_forms(p[idx_i], q[idx_i], domain, boundaries[i])
    return bcs


def solve_pressure(V, compartments, boundaries, params, subboundaries=None, source=None):
    p = TrialFunctions(V)
    q = TestFunction(V)
    K, G = to_constant(params, "hydraulic_conductivity", "convective_fluid_transfer")

    F = pressure_variational_form(p, q, compartments, K, G, source=source)

    if subboundaries is None:
        domain = Domain(V.mesh(), None, None)
    else:
        mesh = V.mesh()
        boundary_tags = MeshFunction("size_t", mesh, 1, 0)
        [subd.mark(boundary_tags, tag) for tag, subd in subboundaries.items()]
        domain = Domain(mesh, None, boundary_tags)


    bcs = process_boundaries_multicompartment(p, q, F, boundaries, V, compartments, domain)

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


def solve_solute(
    c0, p, dt, T, domain, V, compartments, boundaries, params, results_path="../results/concentration/"
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

    # TODO: Collect variational form into function
    F = 0.0
    C0 = project(c0, V)
    for idx_j, j in enumerate(compartments):
        F += (c[idx_j] - C0[idx_j]) * w[idx_j] * dx
        F += (
            dt
            * (inner(D[j] * grad(c[idx_j]) + K[j] / phi[j] * c[idx_j] * grad(p[idx_j]), grad(w[idx_j])))
            * dx
        )
        sj = 0.0
        for idx_i, i in enumerate(compartments):
            if idx_i == idx_j:
                continue
            sj += (
                L[(i, j)] * (c[idx_i] - c[idx_j])
                + G[(i, j)] * (p[idx_i] - p[idx_j]) * (0.5 * (c[idx_i] + c[idx_j]))
            ) / phi[j]
        F -= dt * sj * w[idx_j] * dx

    bcs = []
    for idx_i, i in enumerate(compartments):
        bcs.extend(process_dirichlet(domain, V.sub(idx_i), boundaries[i]))
        F -= dt * process_boundary_forms(c[idx_i], w[idx_i], domain, boundaries[i])
    a = lhs(F)
    l = rhs(F)
    A = assemble(a)

    storage = TimeSeriesStorage("w", results_path, mesh=domain.mesh, V=V)
    t = Constant(0.0)
    storage.write(C0, t)
    mass = np.nan * np.zeros(int(T / dt) + 1)
    idx = 0
    mass[idx] = total_mass(C0, phi, compartments)
    C = Function(V, name="concentration")
    while float(t) < T:
        print_progress(float(t), T)
        A = assemble(a)
        b = assemble(l)
        for bc in bcs:
            bc.apply(A, b)
            solve(A, C.vector(), b)
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

    from multirat.meshprocessing import Domain
    from multirat.parameters import multicompartment_parameters
    from multirat.boundary import DirichletBoundary, RobinBoundary

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
