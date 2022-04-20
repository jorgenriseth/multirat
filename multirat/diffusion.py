import skallaflow as sf
from .computers import BaseComputer


def print_progress(time):
    progress = int(20 * time.t / time.T)
    print(f"[{'=' * progress}{' ' * (20 - progress)}] {time.t / 60:>6.1f}min / {time.T / 60:<5.1f}min", end='\r')


def solve_diffusion_problem(problem, results_path, computer=None):
    if computer is None:
        computer = BaseComputer({})

    problem.init_solver()
    storage = sf.TimeSeriesStorage("w", results_path, mesh=problem.domain.mesh, V=problem.V)
    storage.write(problem.u, problem.time.t)
    computer.initiate(problem)

    problem.time.progress()
    while problem.time.is_valid():
        print_progress(problem.time)

        problem.pre_solve()  # Possible updates to be performed before solving the system.
        problem.solve()

        # print(problem.time.t)
        storage.write(problem.u, problem.time.t)
        computer.compute(problem)

        problem.post_solve()
        problem.time.progress()

    storage.close()
    print()
