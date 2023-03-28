import time as pytime
import  subprocess

from dolfin import FunctionSpace

ENDTIME = 3600.0 * 6.0
resolutions = [8, 16, 32, 64]
resolutions = [64]
models = ["homogeneous", "conservation", "decay"]
timesteps = [60, 600, 3600]


if __name__ == "__main__":
    T = ENDTIME

    print(
        f"""{40*"="}
    Resolutions: \t{resolutions}
    Models: \t{models}
    dt: \t\t{timesteps}
    T: \t\t{T}s i.e. {T/3600:.1f}h
    {40*"="}"""
    )

    proceed = input("Are these setting correct? (y/N)")
    print()
    if proceed.lower() != "y":
        raise RuntimeError("Stop right there!")

    for ncomp in [3, 4, 7]:
        for res in resolutions:
            for model in models:
                for dt in timesteps:
                    tic = pytime.time()
                    print(f"== Mesh {res} | Model: {model} | Timestep: {dt} ==")
                    # Define problem with storage path
                    subprocess.run(
                        f"OMP_NUM_THREADS=1 mpirun -np 8 python main.py \
                            --meshdir {'mesh/'} \
                            --resolution {res} \
                            --bdry {model} \
                            --timestep {dt} \
                            --endtime {T} \
                            --ncomp {ncomp}",
                        shell=True,
                    )
                    
                    toc = pytime.time()
                    elapsed = toc - tic
                    print()
                    print(f"Solved in {int( elapsed / 60):2d}min {int(elapsed %60):2d}s")
                    print()
