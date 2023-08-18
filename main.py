import logging
from typing import Dict, List, Any, Callable
from pathlib import Path


import dolfin as df
import ufl
from dolfin import inner, grad

import pantarei as pr
from pantarei import solvers
from pantarei import meshprocessing
from pantarei.boundary import process_boundary_forms, BoundaryData, DirichletBoundary
from pantarei.computers import BaseComputer
from pantarei.timekeeper import TimeKeeper
from pantarei.fenicsstorage import FenicsStorage

from multirat.boundary_conditions import (
    TracerODEProblemSolver,
    DecayingSASConcentration,
    ConservedSASConcentration,
)
from multirat.initial_conditions import gaussian_expression
from multirat.parameters import multicompartment_parameters


AbstractForm = pr.forms.AbstractForm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compartment_pressure_form(idx_j, p, q, K, gamma, compartments):
    j = compartments[idx_j]
    sj_q = sum(
        [
            gamma[(i, j)] * (p[idx_i] - p[idx_j]) * q[idx_j]
            for idx_i, i in enumerate(compartments)
            if idx_i != idx_j
        ]
    )
    return K[j] * inner(grad(p[idx_j]), grad(q[idx_j])) - sj_q


def multicompartment_pressure_form(compartments: List[str]) -> AbstractForm:
    def create_fem_form(
        V: df.FunctionSpace,
        coefficients: Dict[str, ufl.Coefficient],
        boundaries: Dict[int, List[BoundaryData]],
    ) -> df.Form:
        dx = df.Measure("dx", domain=V.mesh())
        p = df.TrialFunction(V)
        q = df.TestFunction(V)
        K = coefficients["hydraulic_conductivity"]
        gamma = coefficients["convective_fluid_transfer"]
        return sum(
            [
                compartment_pressure_form(idx_j, p, q, K, gamma, compartments)
                for idx_j, j in enumerate(compartments)
            ]
        ) * dx + process_boundary_forms(p, q, V.mesh(), boundaries)

    return create_fem_form


def multicompartment_solute_form(compartments: List[str]) -> AbstractForm:
    def create_fem_form(
        V: df.FunctionSpace,
        coefficients: Dict[str, Any],
        boundaries: List[BoundaryData],
    ) -> df.Form:
        dx = df.Measure("dx", domain=V.mesh())
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        L = coefficients["diffusive_solute_transfer"]
        G = coefficients["convective_solute_transfer"]
        p = coefficients["pressure"]
        phi = coefficients["porosity"]
        dt = coefficients["dt"]
        u0 = coefficients["u0"]
        D = coefficients["effective_diffusion"]
        K = coefficients["hydraulic_conductivity"]
        return sum(
            [
                compartment_concentration_form(
                    idx_j, u, v, u0, L, G, p, phi, dt, D, K, compartments
                )
                for idx_j, j in enumerate(compartments)
            ]
        ) * dx + dt * process_boundary_forms(u, v, V.mesh(), boundaries)

    return create_fem_form


def compartment_concentration_form(
    idx_j, u, v, u0, L, G, p, phi, dt, D, K, compartments
):
    j = compartments[idx_j]
    diffusive_transfer = sum(
        [
            L[(i, j)] * (u[idx_i] - u[idx_j])
            for idx_i, i in enumerate(compartments)
            if idx_i != idx_j
        ]
    )
    convective_transfer = 0.5 * sum(
        [
            0.5 * G[(i, j)] * (p[idx_i] - p[idx_j]) * (u[idx_i] + u[idx_j])
            for idx_i, i in enumerate(compartments)
            if idx_i != idx_j
        ]
    )
    rj = (convective_transfer + diffusive_transfer) / phi[j]
    return (u[idx_j] - u0[idx_j] - dt * rj) * v[idx_j] + dt * inner(
        D[j] * grad(u[idx_j]) + K[j] / phi[j] * u[idx_j] * grad(p[idx_j]),
        grad(v[idx_j]),
    )


def load_mesh(meshdir: Path, n: int):
    # domain = MMSDomain(n)
    domain = meshprocessing.hdf2fenics(meshdir / f"mesh{n}.h5", pack=True)

    # path = Path("~/parkrec/DATA/PAT_002/FENICS/mesh16.h5").expanduser()
    # domain = meshprocessing.hdf2fenics(path, pack=True)
    return domain


def solve_pressure_problem(domain, compartments, coefficients):
    element = df.MixedElement(
        [df.FiniteElement("CG", ufl.Cell(domain.cell_name()), degree=1)]
        * len(compartments)
    )
    p_bdry = coefficients["pressure_boundaries"]
    # TODO: Update to match original problem, with RobinBoundaries
    bdry_data = {
        idx_j: [pr.boundary.DirichletBoundary(df.Constant(p_bdry[j]), "everywhere")]
        for idx_j, j in enumerate(compartments)
    }
    P = solvers.solve_stationary(
        domain=domain,
        element=element,
        coefficients=coefficients,
        form=multicompartment_pressure_form(compartments),
        boundaries=pr.boundary.indexed_boundary_conditions(bdry_data),
        solver=solvers.StationaryProblemSolver("lu", "none"),
        name="pressure",
    )
    return P


def injection(compartments, center, std, phi, **kwargs):
    porosity_total = sum([phi[compartment] for compartment in compartments])

    def load_initial_condition(V, boundaries):
        u0 = {}
        for idx, compartment in enumerate(compartments):
            u0_ = gaussian_expression(center=center, std=std, **kwargs)
            u0_ = df.project(u0_, V.sub(idx).collapse())
            bcs = [
                df.DirichletBC(
                    V.sub(idx).collapse(), boundaries[idx].bc.uD, "on_boundary"
                )
            ]
            u0_ = pr.projectors.smoothing_projector(0.1)(
                u0_, V.sub(idx).collapse(), bcs
            )
            u0_ = pr.utils.rescale_function(u0_, 1.0 / porosity_total)
            u0[compartment] = u0_
        return pr.utils.assign_mixed_function(u0, V, compartments)

    return load_initial_condition


def ecs_injection(compartments, center, std, phi, **kwargs):
    def load_initial_condition(V, boundaries):
        u = {
            "ecs": gaussian_expression(center=center, std=std, **kwargs),
            **{
                compartment: df.Constant(0.0)
                for compartment in compartments
                if compartment != "ecs"
            },
        }
        bcs = pr.boundary.process_dirichlet(V, V.mesh(), boundaries)
        U = pr.projectors.mixed_space_projector(
            compartments, pr.projectors.smoothing_projector(0.05)
        )(u, V, bcs)
        pr.utils.rescale_function(U.sub(0), 1.0 / phi["ecs"])
        return U

    return load_initial_condition


def solve_concentration_problem(
    domain: df.Mesh,
    coefficients: Dict[str, Any],
    compartments: List[str],
    bdry_data: pr.boundary.BoundaryData,
    time: TimeKeeper,
    store: FenicsStorage,
):
    element = df.MixedElement(
        [df.FiniteElement("CG", ufl.Cell(domain.cell_name()), degree=1)]
        * len(compartments)
    )

    boundaries = pr.boundary.indexed_boundary_conditions(bdry_data)
    solute_form = multicompartment_solute_form(compartments)

    # TODO: Horrible reading, move elsewhere.
    def compartment_mass(j, idx_j, phi_j):
        return lambda u: phi_j * df.assemble(u[idx_j] * df.dx)

    def sas_concentration(u):
        return float(coefficients["sas_concentration"])

    phi = coefficients["porosity"]
    computer = BaseComputer(
        {
            **{
                f"mass_{j}": compartment_mass(j, idx_j, phi[j])
                for idx_j, j in enumerate(compartments)
            },
            "mass_sas": lambda u: coefficients["csf_volume"]
            * float(coefficients["sas_concentration"]),
        }
    )
    stationary_solver = solvers.StationaryProblemSolver("lu", "none")
    time.reset()
    results = solvers.solve_time_dependent(
        domain=domain,
        element=element,
        coefficients=coefficients,
        form=solute_form,
        boundaries=boundaries,
        initial_condition=ecs_injection(
            compartments, [3.0, 3.0, 3.0], 1.0, phi, degree=2
        ),
        time=time,
        solver=TracerODEProblemSolver(
            stationary_solver, coefficients["sas_concentration"]
        ),
        storage=store,
        name="concentrations",
        computer=computer,
        # projector=projector,
    )
    return results


def multicompartment_model(
    meshdir: Path,
    resolution: int,
    bdry_type: str,
    compartments: List[str], 
    timestep: float,
    endtime: float,
    results_path: Path,
    parameter_loader: Callable[..., Dict[str, Any]] = multicompartment_parameters,
):
    domain = load_mesh(Path(meshdir), resolution)
    coefficients = parameter_loader(compartments)
    coefficients["csf_renewal_rate"] = 0.1
    coefficients["csf_volume_fraction"] = 0.12
    coefficients["csf_volume"] = coefficients["csf_volume_fraction"] * df.assemble(
        1.0 * df.Measure("dx", domain=domain)
    )

    ncomp = len(compartments)
    results_path = Path(
        f"results/multicompartment-{resolution}-{ncomp}comp-{bdry_type}"
    )
    store = pr.fenicsstorage.FenicsStorage(results_path / "results.hdf", "w")

    logging.info("Solving pressure problem")
    P = solve_pressure_problem(domain, compartments, coefficients)
    store.write_function(P, "pressure")
    coefficients["pressure"] = P

    time = TimeKeeper(timestep, endtime)
    coefficients["dt"] = timestep
    coefficients["sas_concentration"] = setup_sas_concentration(
        bdry_type, domain, compartments, coefficients, time
    )
    bdry_data = {
        idx_j: [DirichletBoundary(coefficients["sas_concentration"], "everywhere")]
        for idx_j, _ in enumerate(compartments)
    }

    logging.info("Solving concentration problem")
    results = solve_concentration_problem(
        domain, coefficients, compartments, bdry_data, time, store
    )
    store.close()

    logging.info("Converting to XDMF")
    store = pr.fenicsstorage.FenicsStorage(results_path / "results.hdf", "r")
    store.to_xdmf("concentrations", compartments)
    store.to_xdmf("pressure", compartments)
    store.close()
    return results


def setup_sas_concentration(bdry_type: str, domain, compartments, coefficients, time):
    if bdry_type.lower() == "homogeneous":
        return df.Constant(0.0)
    elif bdry_type == "conservation":
        return ConservedSASConcentration(domain, coefficients, compartments, time)
    elif bdry_type == "decay":
        return DecayingSASConcentration(domain, coefficients, compartments, time)

    raise ValueError(f"Unknown boundary type: {bdry_type}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meshdir",
        help="Mesh directory"
    )
    parser.add_argument("--resolution", type=int, help="Mesh resolution")
    parser.add_argument(
        "--bdry", help="Boundary type (homogeneous, conservation, decay)"
    )
    parser.add_argument("--timestep", type=float, default=60.0, help="Timestep")
    parser.add_argument("--endtime", type=float, default=3600.0 * 60.0, help="Timestep")
    parser.add_argument("--ncomp", type=int, default=3, help="Number of compartments")
    parser.add_argument("--resultprefix", help="Prefix for result files", default="multicompartment")
    args = parser.parse_args()

    if args.ncomp == 3:
        compartments = ["ecs", "pvs_arteries", "pvs_veins"]
    elif args.ncomp == 4:
        compartments = ["ecs", "pvs_arteries", "pvs_capillaries", "pvs_veins"]
    elif args.ncomp == 7:
        compartments = ["ecs", "pvs_arteries", "pvs_capillaries", "pvs_veins", "arteries", "capillaries", "veins"]
    else:
        raise ValueError(f"Nonstandard number of compartments: {args.ncomp}")

    results_path = Path(
        f"results/{args.resultprefix}-{args.resolution}-{args.ncomp}comp-{args.bdry}"
    )
    results_path.mkdir(parents=True, exist_ok=True)
    import time
    tic = time.time()
    results = multicompartment_model(
        meshdir=args.meshdir,
        resolution=args.resolution,
        bdry_type=args.bdry,
        compartments=compartments,
        timestep=args.timestep,
        endtime=args.endtime,
        results_path=results_path,
    )
    toc = time.time()
    print()
    print(f"Elapsed time: {toc - tic:.2f} s")
    
    if df.MPI.comm_world.rank == 0:
        import matplotlib.pyplot as plt
        import numpy as np
        times = TimeKeeper(args.timestep, args.endtime).as_vector()
        X = np.zeros((len(times), len(results.values)+1))
        X[:, 0] = times
        for i, values in enumerate(results.values.values()):
            X[:, i+1] = values
        np.savetxt(
            results_path / "computed.txt",
            X,
            delimiter=" ",
            header=" ".join(["times", *results.values.keys()])
        )
        total_mass = sum(
            results[f"mass_{j}"] for j in ["ecs", "pvs_arteries", "pvs_veins"]
        )
        for key, values in results.values.items():
            if key.startswith("mass_"):
                plt.plot(values, label=key)
        plt.plot(total_mass, label="total mass")
        plt.plot(total_mass + results["mass_sas"], label="total mass + sas")
        plt.legend()

        logger.info("Plotting")
        plt.show()
