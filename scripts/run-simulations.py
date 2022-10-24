import numpy as np
import matplotlib.pyplot as plt
import time as pytime

from dolfin import *
from multirat import *
from multirat.parameters import PARAMS
import multirat.base.projectors as projectors


ENDTIME = 3600. * 6. 
resolutions = [8, 16, 32, 64]
resolutions = [64]
models = ['homogeneous', 'tracerconservation', 'tracerdecay']
timesteps = [60, 600, 3600]
CASES = {
    "homogeneous": {
        "problem": HomogeneousProblem,
        "label": "Homogeneous",
    },
    "tracerconservation" : {
        "problem": TracerConservationProblem,
        "label": "Tracer Conservation"
    },
    "tracerdecay": {
        "problem": TracerDecayProblem,
        "label": "Tracer Decay"
    }
}

def ratbrain_diffusion(meshfile, results_path, model, timestep, endtime, computer=None,
                       center=PARAMS["injection_center"], std = PARAMS["injection_spread"], degree=1):    
    # Load functionspace and domain 
    domain = hdf2fenics(meshfile, pack=True)
    V = FunctionSpace(domain.mesh, "CG", degree)

    # Define initial condition and rescale to unit mass
    u0_expression = gaussian_expression(center=center, std=std, degree=degree)
    u0 = HomogeneousDirichletProjector().project(u0_expression, V)
    projectors.rescale_function(u0, 1.)  # Scale to unit mass total
    

    # Define timekeeper
    time = TimeKeeper(timestep, endtime)
        
    # Initiate problem, computers, and solve the problem
    problem = CASES[model]["problem"](domain, time, u0, V)
    
    # Solve
    solve_diffusion_problem(problem, results_path, computer)
    problem.time.reset()
    
    return problem


if __name__ == "__main__":
    T = ENDTIME

    print(f"""{40*"="}
    Resolutions: \t{resolutions}
    Models: \t{models}
    dt: \t\t{timesteps}
    T: \t\t{T}s i.e. {T/3600:.1f}h
    {40*"="}""")

    proceed = input("Are these setting correct? (y/N)")
    print()
    if proceed.lower() != 'y':
        raise RuntimeError("Stop right there!")
        
    for res in resolutions:
        for model in models:
            for dt in timesteps:
                tic = pytime.time()
                print(f"== Mesh {res} | Model: {model} | Timestep: {dt} ==")
                # Define problem with storage path
                path = f"{RESULTS_DIR}/results-{model}-mesh{res}-dt{dt}"
                meshfile = f"{MESH_DIR}/mesh{res}.h5"
                problem = ratbrain_diffusion(meshfile, path, model, dt, T)
                toc = pytime.time()
                elapsed = toc - tic
                print()
                print(f"Solved in {int( elapsed / 60):2d}min {int(elapsed %60):2d}s")
                print()
