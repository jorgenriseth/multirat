import dolfin as df
from dolfin import FacetNormal, Measure, grad, inner
from ufl import Coefficient


class TracerODEProblemSolver:
    def __init__(self, stationarysolver, csf_concentration: Coefficient):
        self.solver = stationarysolver
        self.csf_concentration = csf_concentration

    def solve(self, u, A, b, dirichlet_bcs):
        if isinstance(self.csf_concentration, SASConcentration):
            self.csf_concentration.update(u)
        return self.solver.solve(u, A, b, dirichlet_bcs)


class SASConcentration:
    pass


class ConservedSASConcentration(df.Constant, SASConcentration):
    def __init__(self, domain, coefficients, compartments, time):
        super().__init__(0.0)
        self.compartments = compartments
        self.coefficients = coefficients
        self.time = time
        self.ds = Measure("ds", domain=domain)
        self.n = FacetNormal(domain)
        self.Vcsf = coefficients["csf_volume_fraction"] * df.assemble(
            1.0 * df.Measure("dx", domain=domain)
        )
        self.has_init = False

    def update(self, u0):
        if not self.has_init:
            self.N0 = total_brain_content(u0, self.coefficients["porosity"], self.compartments)
            self.has_init=True
        Q = total_flux(u0, self.coefficients, self.compartments, self.n, self.ds)
        Nnow = total_brain_content(u0, self.coefficients["porosity"], self.compartments)
        self.assign((self.N0 - Nnow + self.time.dt * Q) / self.Vcsf)


class DecayingSASConcentration(df.Constant, SASConcentration):
    def __init__(self, domain, coefficients, compartments, time):
        super().__init__(0.0)
        self.compartments = compartments
        self.coefficients = coefficients
        self.time = time
        self.ds = Measure("ds", domain=domain)
        self.n = FacetNormal(domain)
        self.decay = coefficients["csf_renewal_rate"]
        self.Vcsf = coefficients["csf_volume_fraction"] * df.assemble(
            1.0 * df.Measure("dx", domain=domain)
        )

    def update(self, u0):
        Q = total_flux(u0, self.coefficients, self.compartments, self.n, self.ds)
        self.assign(
            df.exp(-self.decay * self.time.dt) * (self + self.time.dt / self.Vcsf * Q)
        )


def compartment_flux_density(Dj, cj, Kj, phi_j, pj):
    return -phi_j * Dj * grad(cj) - Kj * cj * grad(pj)


def total_flux(C, coefficients, compartments, n, ds):
    Q = 0.0
    for idx, j in enumerate(compartments):
        phi_j = coefficients["porosity"][j]
        Dj = coefficients["effective_diffusion"][j]
        Kj = coefficients["hydraulic_conductivity"][j]
        pj = coefficients["pressure"].sub(idx)
        qj = compartment_flux_density(Dj, C.sub(idx), Kj, phi_j, pj)
        Q += df.assemble(inner(qj, n) * ds)
    return Q


def total_brain_content(u0, phi, compartments):
    return df.assemble(
        sum([phi[j] * u0.sub(idx) for idx, j in enumerate(compartments)]) * df.dx
    )
