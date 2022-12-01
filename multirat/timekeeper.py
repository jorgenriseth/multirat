from numpy import linspace


class TimeKeeper:
    def __init__(self, dt, endtime):
        self.t = 0.0
        self.iter = 0
        self.dt = dt
        self.T = endtime

    def progress(self):
        self.t += self.dt
        self.iter += 1

    def is_valid(self):
        return self.t <= self.T

    def reset(self):
        self.t = 0.0
        self.iter = 0

    def get_vector(self):
        return linspace(0, self.T, len(self))

    def __len__(self):
        return int(self.T // self.dt + 1)
