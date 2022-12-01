from multirat.parameters import multicompartment_parameters, get_base_parameters
from multirat.boundary import DirichletBoundary
from dolfin import Constant, FacetNormal, Measure, assemble, inner, grad, exp


class HomogeneousDirichletBoundary(DirichletBoundary):
    def __init__(self):
        super().__init__(Constant(0.0), "everywhere")


class TracerODEBoundary(DirichletBoundary):
    def __init__(self):
        self.g = Constant(0.0)
        super().__init__(self.g, "everywhere")

        self.k = float()  # = phi * D / Vcsf, to be read from file.
        self.n = None  # Facetnormal
        self.ds = None  # Boundary measure

    def process(self, domain, space):
        self.set_parameters(domain)
        self.n = FacetNormal(domain.mesh)
        self.ds = Measure("ds", domain=domain.mesh, subdomain_data=domain.boundaries)
        return super().process(domain, space)

    def set_parameters(self, domain):
        params = multicompartment_parameters(["ecs"])
        phi = params["porosity"]["ecs"]
        D = params["effective_diffusion"]["ecs"]
        Vbrain = assemble(1.0 * Measure("dx", domain=domain.mesh))
        Vcsf = 0.1 * Vbrain
        self.k = phi * D / Vcsf


class TracerConservationBoundary(TracerODEBoundary):
    def __init__(self):
        super().__init__()

    def update(self, u0, time):
        self.g.assign(self.g - time.dt * self.k * assemble(inner(grad(u0), self.n) * self.ds))


class TracerDecayBoundary(TracerODEBoundary):
    def __init__(self, decay=None):
        if decay is None:
            params = get_base_parameters()
            decay = params["csf_renewal_rate"]
        super().__init__()
        self.decay = decay  # CSF renewal rate

    def update(self, u0, time):
        self.g.assign(
            exp(-self.decay * time.dt)
            * (self.g - time.dt * self.k * assemble(inner(grad(u0), self.n) * self.ds))
        )

