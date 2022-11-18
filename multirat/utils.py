from dolfin import FunctionSpace


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