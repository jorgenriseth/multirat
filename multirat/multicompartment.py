from pathlib import Path

import numpy as np
from dolfin import *

from multirat.boundary import process_boundary_forms, process_dirichlet
from multirat.computers import BaseComputer
from multirat.parameters import to_constant
from multirat.timeseriesstorage import TimeSeriesStorage, DummyStorage


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
                L[(i, j)] * (c[idi] - c[idj]) + G[(i, j)] * (p[idi] - p[idj]) * (0.5 * (c[idi] + c[idj]))
            ) / phi[j]
        F -= dt * sj * w[idj] * dx
        if source is not None:
            F -= dt * source[j] * w[idj] * dx
    return F


def update_time(expr, newtime):
    if hasattr(expr, "t"):
        expr.t.assign(newtime)


class MassComputer(BaseComputer):
    def __init__(self, phi, compartments):
        super().__init__({"total mass": lambda c: total_mass(c, phi, compartments)})


def get_storage(results_path, mesh, V):
    if results_path is not None:
        storage = TimeSeriesStorage("w", results_path, mesh=mesh, V=V)
    return DummyStorage()


def visualize(storage):
    if isinstance(storage, DummyStorage):
        return
    visual = TimeSeriesStorage("r", storage.filepath)
    visual.to_xdmf(compartments)
    visual.close()


def solve_solute(
    c0, p, time, domain, V, compartments, boundaries, params, results_path=None, source=None, computer=None
):
    if computer is None:
        computer = BaseComputer({})
    phi, D, K, G, L = to_constant(
        params,
        "porosity",
        "effective_diffusion",
        "hydraulic_conductivity",
        "convective_solute_transfer",
        "diffusive_solute_transfer",
    )

    c = TrialFunctions(V)
    w = TestFunction(V)
    C0 = project(c0, V)

    F_bdry, bcs = process_boundaries_multicompartment(c, w, boundaries, V, compartments, domain)
    F = solute_variational_form(c, w, compartments, time.dt, C0, p, D, K, L, G, phi, source=source)
    F -= time.dt * F_bdry
    l = rhs(F)
    A = assemble(lhs(F))

    time.reset()
    computer.initiate(time, C0)

    storage = get_storage(results_path, domain.mesh, V)
    storage.write(C0, float(time))
    C = Function(V, name="concentration")
    for _ in range(len(time) - 1):
        time.progress()
        print_progress(float(time), time.endtime)

        C = solve_timestep(A, l, C, bcs)
        C0.assign(C)
        storage.write(C, float(time))
        computer.compute(time, C)
    storage.close()
    visualize(storage)
    print()
