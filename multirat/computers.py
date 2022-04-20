from numpy import zeros

class BaseComputer:
    def __init__(self, function_dict):
        self.functions = function_dict
        self.initiated = False
        self.values = {}

    def initiate(self, problem):
        self.create_value_dict(problem.time)
        self.compute(problem)

    def create_value_dict(self, timekeeper):
        self.values = {key: zeros(len(timekeeper)) for key in self.functions}

    def compute(self, problem):
        for key, function in self.functions.items():
            self.values[key][problem.time.iter] = function(problem.u)

    def __getitem__(self, item):
        return self.values[item]
