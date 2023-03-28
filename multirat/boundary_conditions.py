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

    def update(self, u0):
        q = total_flux_density(
            u0, self.coefficients["pressure"], self.coefficients, self.compartments
        )
        Q = df.assemble(inner(q, self.n) * self.ds)
        self.assign(self + self.time.dt / self.Vcsf * Q)

        # phi = self.coefficients["porosity"]
        # N0 = 1.0
        # content = total_brain_content(u0, phi, self.compartments)
        # self.assign((N0 - content) + self.time.dt / self.Vcsf * Q)


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
        q = total_flux_density(
            u0, self.coefficients["pressure"], self.coefficients, self.compartments
        )
        Q = df.assemble(inner(q, self.n) * self.ds)
        self.assign(
            df.exp(-self.decay * self.time.dt) * (self + self.time.dt / self.Vcsf * Q)
        )


def compartment_flux_density(Dj, cj, Kj, phi_j, pj):
    return -Dj * grad(cj) - Kj / phi_j * cj * grad(pj)


def total_flux_density(C, P, coefficients, compartments):
    D, K, phi = (
        coefficients[param]
        for param in ["effective_diffusion", "hydraulic_conductivity", "porosity"]
    )
    return sum(
        [
            phi[j] * compartment_flux_density(D[j], C[idx_j], K[j], phi[j], P[idx_j])
            for idx_j, j in enumerate(compartments)
        ]
    )


def total_brain_content(u0, phi, compartments):
    return df.assemble(
        sum([phi[j] * u0.sub(idx) for idx, j in enumerate(compartments)]) * df.dx
    )
