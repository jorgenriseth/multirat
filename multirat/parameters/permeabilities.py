from numpy import exp, sqrt, pi
from typing import Union, Dict
from pint import Quantity, UnitRegistry

from multirat.parameters.base import SHARED_PARAMETERS, unpack_shared_parameter, mm, s


def unpack_membrane_params(membrane_parameters: Dict[str, Union[str, Dict]]):
    unpacked = {}
    for (
        param,
        value_dict,
    ) in membrane_parameters.items():  # loop through different membrane parameters.
        unpacked_param = {}
        if isinstance(value_dict, dict):
            unpacked_param = unpack_shared_parameter(
                value_dict, SHARED_PARAMETERS["blood"]
            )
        else:
            unpacked[
                param
            ] = value_dict  #  Not sure if ever occurs, but kept for safety

        unpacked[param] = unpacked_param
    return unpacked


def diffusive_transfer_coefficients(params, solute):
    P = diffusive_permeabilities(params, solute)
    surface_volume_ratio = params["surface_volume_ratio"]
    transfer = {}
    for vi in SHARED_PARAMETERS["blood"]:
        transfer[("ecs", f"pvs_{vi}")] = (
            P[("ecs", f"pvs_{vi}")] * surface_volume_ratio[("ecs", vi)]
        )
        transfer[(vi, f"pvs_{vi}")] = (
            P[(vi, f"pvs_{vi}")] * surface_volume_ratio[("ecs", vi)]
        )
    return {key: val.to(1 / (s)) for key, val in transfer.items()}


def diffusive_permeabilities(params, solute):
    bbb_layers = [
        "glycocalyx",
        "inner_endothelial_cleft",
        "endothelial_junction",
        "outer_endothelial_cleft",
        "basement_membrane",
    ]
    P = {}
    for vi in SHARED_PARAMETERS["blood"]:
        dvi = params["diameter"][vi]
        Rvi = diffusive_resistance_membrane_layer(params, solute, vi)
        R_bbb = sum([Rvi[layer] for layer in bbb_layers])
        R_aef = Rvi[f"aef"]
        P[(vi, f"pvs_{vi}")] = (
            1.0 / (pi * dvi) / R_bbb if solute != "inulin" else 0.0 * mm / s
        )
        P[("ecs", f"pvs_{vi}")] = 1.0 / (pi * dvi) / R_aef
    return P


def diffusive_resistance_membrane_layer(params, solute, vessel):
    membranes = unpack_membrane_params(params["membranes"])
    D_free = params["diffusion_coefficient_free"][solute]
    solute_radius = params["solute_radius"][solute]
    R = {}
    for layer, layertype in membranes["layertype"].items():
        if layertype == "fiber":
            Vf = membranes["fiber_volume_fraction"][layer][vessel]
            r = membranes["elementary_radius"][layer][vessel]
            D_eff = diffusion_fibrous(D_free, solute_radius, r, Vf)
        elif layertype == "pore":
            r = membranes["elementary_radius"][layer][vessel]
            D_eff = diffusion_porous(D_free, solute_radius, r)
        else:
            raise ValueError(f"layertype should be 'pore' or 'fiber', got {layertype}")

        R[layer] = solute_resistance_layer(
            membranes["thickness"][layer][vessel], r, D_eff
        )
    return R


def diffusion_fibrous(D_free, solute_radius, fiber_radius, fiber_volume_fraction):
    return D_free * exp(
        -sqrt(fiber_volume_fraction) * (1.0 + solute_radius / fiber_radius)
    )


def diffusion_porous(
    D_free: Quantity, solute_radius: Quantity, pore_radius: Quantity
) -> Quantity:
    beta = solute_radius / pore_radius
    return D_free * (
        1.0
        - 2.10444 * beta  # **6
        + 2.08877 * beta**3
        - 0.094813 * beta**5
        - 1.372 * beta**6
    )


def solute_resistance_layer(layer_thickness, elementary_radius, effective_diffusion):
    return layer_thickness / (2.0 * elementary_radius * effective_diffusion)


if __name__ == "__main__":
    from multirat.parameters.base import get_base_parameters, print_quantities
    print_quantities(
        diffusive_resistance_membrane_layer(
            get_base_parameters(), "amyloid_beta", "arteries"
        )
    )
    print()
    print_quantities(diffusive_transfer_coefficients(get_base_parameters(), "inulin"))
