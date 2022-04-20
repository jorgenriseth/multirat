from .parameters import PARAMS
import skallaflow as sf
from dolfin import Constant, FacetNormal, Measure, assemble, inner, grad, exp


class HomogeneousDirichletBoundary(sf.DirichletBoundary):
    def __init__(self):
        super().__init__(Constant(0.), "everywhere")


# TODO: Add scaling directly in parameters-file
class TracerODEBoundary(sf.DirichletBoundary):
    def __init__(self):
        self.g = Constant(0.)
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
        phi = PARAMS["porosity_ecs"]
        D = PARAMS["diffusion_constant"]
        Vbrain = assemble(1. * Measure('dx', domain=domain.mesh))
        Vcsf = 0.1 * Vbrain
        self.k = phi * D / Vcsf


class TracerConservationBoundary(TracerODEBoundary):
    def __init__(self):
        super().__init__()

    def update(self, u0, time):
        self.g.assign(self.g - time.dt * self.k * assemble(inner(grad(u0), self.n) * self.ds))


class TracerDecayBoundary(TracerODEBoundary):
    def __init__(self, decay=PARAMS["decay"]):
        super().__init__()
        self.decay = decay  # CSF renewal rate

    def update(self, u0, time):
        self.g.assign(
            exp(-self.decay * time.dt) * (self.g - time.dt * self.k * assemble(inner(grad(u0), self.n) * self.ds))
        )

