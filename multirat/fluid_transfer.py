from multirat.parameters.base import (
    SHARED_PARAMETERS,
    print_quantities,
    Pa,
    s,
)


def effective_diffusion(D_free, tortuosity):
    return D_free / tortuosity**2


def permeability(viscosity, resistance, length_area_ratio):
    return viscosity * length_area_ratio / resistance


def compute_partial_fluid_transfer(brain_volume, resistance, total_transfer):
    new_resistance = 1.0 / (total_transfer * brain_volume) - resistance
    return 1.0 / (new_resistance * brain_volume)


def compute_connected_fluid_transfer(brain_volume, flow_rate, pressure_drop):
    return flow_rate / (pressure_drop * brain_volume)


def compute_length_area_ratio(p):
    kappa = p["permeability"]["ecs"]
    R = p["resistance"]["ecs"]
    mu = p["resistance"]["ecs"]
    return {"length_area_ratio": kappa * R / mu}


def compute_convective_fluid_transfer(params):
    T = {}
    V = params["human_brain_volume"]

    # Compute membrane transfer coefficients.
    # FIXME: Apparent error in resistance ecs-capillaries.
    for vi in SHARED_PARAMETERS["blood"]:
        L_ecs_vi = params["hydraulic_conductivity"][("ecs", vi)]
        surface_ratio = params["surface_volume_ratio"][("ecs", vi)]
        T_ecs_vi = L_ecs_vi * surface_ratio
        R_ecs_vi = params["resistance"][("ecs", vi)]

        T[("ecs", f"pvs_{vi}")] = 1.0 / (V * R_ecs_vi)
        T[(f"pvs_{vi}", vi)] = compute_partial_fluid_transfer(V, R_ecs_vi, T_ecs_vi)

    # Compute connected transfer coefficients.
    connected = [("arteries", "capillaries"), ("capillaries", "veins")]
    for vi, vj in connected:
        Q = params["flowrate"]
        dp = params["pressure_drop"]
        T[(vi, vj)] = compute_connected_fluid_transfer(V, Q["blood"], dp[(vi, vj)])
        T[(f"pvs_{vi}", f"pvs_{vj}")] = compute_connected_fluid_transfer(
            V, Q["csf"], dp[(f"pvs_{vi}", f"pvs_{vj}")]
        )
    return {
        key: val.to(1/(Pa * s)) for key, val in T.items()
    }


if __name__ == "__main__":
    from multirat.parameters.base import get_base_parameters, print_quantities
    p = get_base_parameters()
    print_quantities(compute_convective_fluid_transfer(p))
