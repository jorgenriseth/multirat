def set_mms_boundary(
    u_sympy: sp.Expr,
    domain: MMSDomain,
    boundary: BoundaryData,
    coefficients: Dict[str, Coefficient],
    degree: int,
):
    dim = domain.topology.dim()
    u = mms_placeholder(dim)
    if isinstance(boundary, DirichletBoundary):
        uD = ulfy.Expression(u, subs={u: u_sympy}, degree=degree)
        return DirichletBoundary(uD, boundary.tag)

    if isinstance(boundary, NeumannBoundary):
        D = coefficients["D"]
        n = domain.normals[boundary.tag]
        g = ulfy.Expression(inner(-D * grad(u), n), subs={u: u_sympy}, degree=degree)
        return NeumannBoundary(g, boundary.tag)

    if isinstance(boundary, RobinBoundary):
        D = coefficients["D"]
        n = domain.normals[boundary.tag]
        g = ulfy.Expression(u + (D / boundary.a) * inner(grad(u), n), subs={u: u_sympy}, degree=degree)
        return RobinBoundary(boundary.a, g, boundary.tag)

    raise NotImplementedError(f"MMS expression for boundary-type {type(boundary)} not implemented.")
