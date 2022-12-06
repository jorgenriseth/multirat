from multirat.computers import BaseComputer
from multirat.problems import BaseDiffusionProblem
from multirat.timeseriesstorage import TimeSeriesStorage


def print_progress(time):
    progress = int(20 * time.t / time.T)
    print(
        f"[{'=' * progress}{' ' * (20 - progress)}] {time.t / 60:>6.1f}min / {time.T / 60:<5.1f}min", end="\r"
    )


def solve_diffusion_problem(problem: BaseDiffusionProblem, results_path, computer=None):
    if computer is None:
        computer = BaseComputer({})

    problem.init_solver()
    storage = TimeSeriesStorage("w", results_path, mesh=problem.domain.mesh, V=problem.V)
    storage.write(problem.u, problem.time.t)
    computer.initiate(problem)

    problem.time.progress()
    while problem.time.is_valid():
        print_progress(problem.time)

        problem.pre_solve()  # Possible updates to be performed before solving the system.
        problem.solve()

        storage.write(problem.u, problem.time.t)
        computer.compute(problem)

        problem.post_solve()
        problem.time.progress()

    storage.close()
    print()
    return computer
