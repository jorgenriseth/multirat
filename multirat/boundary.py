from abc import ABC, abstractmethod
from typing import Union, List, TypeAlias

from dolfin import FacetNormal, Measure, TrialFunction, TestFunction, FunctionSpace, Form
from fenics import DirichletBC
from ufl import inner, Coefficient
from ufl.indexed import Indexed

from multirat.meshprocessing import Domain


class BoundaryData:
    def __init__(self, condition_type: str, tag: Union[int, str], boundary_name: str = None):
        self.type = condition_type
        self.tag = tag
        if boundary_name is None:
            self.name = "boundary_" + str(tag)
        else:
            self.name = boundary_name


class DirichletBoundary(BoundaryData):
    def __init__(self, value: Coefficient, tag: Union[int, str]):
        self.g = value
        super().__init__("Dirichlet", tag=tag)

    def process(self, V: FunctionSpace, domain: Domain) -> DirichletBC:
        if self.tag == "everywhere":
            return DirichletBC(V, self.g, "on_boundary")
        return DirichletBC(V, self.g, domain.boundaries, self.tag)


class IndexedDirichletBoundary(DirichletBoundary):
    def __init__(self, idx: int, value: Coefficient, tag: Union[int, str]):
        super().__init__(value, tag)
        self.idx = idx

    def process(self, V: FunctionSpace, domain: Domain) -> DirichletBC:
        super().process(V.sub(self.idx), domain)


class VariationalBoundary(BoundaryData):
    @abstractmethod
    def variational_boundary_form(
        self, u: Union[TrialFunction, Indexed], v: Union[TestFunction, Indexed], n: FacetNormal, ds: Measure
    ) -> Form:
        pass

    def process(
        self, u: Union[TrialFunction, Indexed], v: Union[TestFunction, Indexed], domain: Domain
    ) -> Form:
        n = FacetNormal(domain.mesh)
        if domain.boundaries is None:
            ds = Measure("ds", domain=domain.mesh)
        else:
            ds = Measure("ds", domain=domain.mesh, subdomain_data=domain.boundaries)
        return self.variational_boundary_form(u, v, n, ds)


class IndexedVariationalBoundary(VariationalBoundary):
    def __init__(self, idx: int, value: Coefficient, tag: Union[int, str]):
        super().__init__(value, tag)
        self.idx = idx

    def process(
        self, u: Indexed, v: Indexed, domain: Domain
    ) -> Form:
        super().__init__(u[self.idx], v[self.idx], )


class NeumannBoundary(VariationalBoundary):
    def __init__(self, value: Coefficient, tag: Union[int, str], **kwargs):
        self.g = value
        super().__init__("Neumann", tag=tag, **kwargs)

    def variational_boundary_form(
        self, _: Union[TrialFunction, Indexed], v: Union[TestFunction, Indexed], n: FacetNormal, ds: Measure
    ) -> Form:
        return inner(self.g, v) * ds(self.tag)


class RobinBoundary(VariationalBoundary):
    def __init__(self, coeff, value, tag, **kwargs):
        self.a = coeff
        self.g = value
        super().__init__("Robin", tag=tag, **kwargs)

    def variational_boundary_form(
        self, u: Union[TrialFunction, Indexed], v: Union[TestFunction, Indexed], _: FacetNormal, ds: Measure
    ) -> Form:
        return self.a * (self.g - u) * v * ds(self.tag)


class BoundaryConditions:
    def __init__(self, boundaries: List[BoundaryData]):
        self._boundaries = boundaries

    def process(self, V: FunctionSpace, domain: Domain) -> List[DirichletBC]:
        return [bc.process(V, domain) for bc in self._boundaries if isinstance(bc, DirichletBoundary)]

    def process(
        self, u: Union[TrialFunction, Indexed], v: Union[TestFunction, Indexed], domain: Domain
    ) -> Form:
        return sum(
            [bc.process(u, v, domain) for bc in self._boundaries if isinstance(bc, VariationalBoundary)]
        )


def process_dirichlet(domain, space, boundaries):
    return [bc.process(domain, space) for bc in boundaries if isinstance(bc, DirichletBoundary)]


def process_boundary_forms(trial, test, domain, boundaries):
    return sum([bc.process(trial, test, domain) for bc in boundaries if isinstance(bc, VariationalBoundary)])
