from dolfin import Constant
from numpy import linspace, ceil


class TimeKeeper(Constant):
    def __init__(self, dt, endtime):
        self.iter = 0
        self.dt = dt
        self.endtime = endtime
        super().__init__(0.0)

    def progress(self):
        self.assign(self + self.dt)
        self.iter += 1

    def is_valid(self):
        return float(self.t) <= self.endtime

    def reset(self):
        self.assign(0.0)
        self.iter = 0

    def get_vector(self):
        return linspace(0, (len(self) - 1) * self.dt, len(self))

    def __len__(self):
        return int(ceil(self.endtime / self.dt) + 1)
