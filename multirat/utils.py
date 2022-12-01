from dolfin import Function, FunctionAssigner, FunctionSpace, interpolate


class LabeledFunctionSpace(FunctionSpace):
    def __init__(self, mesh, element, labels):
        super().__init__(mesh, element)
        self.dim = len(labels)
        self.idx_map = {label: idx for idx, label in enumerate(labels)}
        if not self.num_sub_spaces() == len(labels):
            raise ValueError("V should be MixedElement space of same dimension as the number of labels.")

    def __getitem__(self, label):
        return self.sub(self.idx_map[label])

    def __iter__(self):
        self.iter = -1
        return self

    def __next__(self):
        self.iter += 1
        if self.iter >= self.dim:
            raise StopIteration
        return self.sub(self.iter)


def assign_mixed_function(p, V, compartments):
    """Create a function in a mixed function-space with sub-function being
    assigned from a dictionray of functions living in the subspaces."""
    P = Function(V)
    subspaces = [V.sub(idx).collapse() for idx, _ in enumerate(compartments)]
    Pint = [interpolate(p[j], Vj) for j, Vj in zip(compartments, subspaces)]
    assigner = FunctionAssigner(V, subspaces)
    assigner.assign(P, Pint)
    return P
