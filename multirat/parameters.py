from multirat.parameters import *
from itertools import combinations

import numbers

from typing import Dict, List
from pint import Quantity, UnitRegistry

# Define units
ureg = UnitRegistry()
mmHg = ureg.mmHg
Pa = ureg.Pa
m = ureg.m
mm = ureg.mm
um = ureg.um
nm = ureg.nm
s = ureg.s
minute = ureg.min
mL = ureg.mL


SHARED_PARAMETERS = {
    "pvs": ["pvs_arteries", "pvs_capillaries", "pvs_veins"],
    "csf": ["ecs", "pvs_arteries", "pvs_capillaries", "pvs_veins"],
    "blood": ["arteries", "capillaries", "veins"],
    "large_vessels": ["arteries", "veins"],
    "all": [
        "ecs",
        "pvs_arteries",
        "pvs_capillaries",
        "pvs_veins",
        "arteries",
        "capillaries",
        "veins",
    ],
    "bbb": [
        ("pvs_arteries", "arteries"),
        ("pvs_capillaries", "capillaries"),
        ("pvs_veins", "veins"),
    ],
    "aef": [("ecs", "pvs_arteries"), ("ecs", "pvs_capillaries"), ("ecs", "pvs_veins")],
}

BASE_PARAMETERS = {
    "brain_volume": 2450.0 * mm**3,
    "human_brain_volume": 1.0e6 * mm**3,
    "diffusion_coefficient_free": {
        "inulin": 2.98 * mm**2 / s,
        "amyloid_beta": 1.8 * mm**2 / s,
    },
    "boundary_pressure": {
        "arteries": 120.0 * mmHg,
        "veins": 7.0 * mmHg,
        "pvs_arteries": 4.74 * mmHg,
        "pvs_veins": 3.36 * mmHg,
        "ecs": 3.74 * mmHg,
    },
    "tortuosity": 1.7,
    "osmotic_pressure": {
        "blood": 20.0 * mmHg,
    },
    "osmotic_pressure_fraction": {
        "csf": 0.2,  # # Osm. press. computed as this constat * osmotic_pressure-blood
    },
    "osmotic_reflection": {
        "all": 0.2
    },
    "porosity": {
        "ecs": 0.14,
    },
    "vasculature_volume_fraction": 0.0329,
    "vasculature_fraction": {
        "arteries": 0.2,
        "capillaries": 0.1,
        "veins": 0.3,
    },
    "pvs_volume_fraction": 0.003,
    "viscosity": {
        "blood": 2.67e-3 * Pa * s,
        "csf": 7.0e-4 * Pa * s,
    },
    "permeability": {"ecs": 2.0e-11 * mm**2},
    "hydraulic_conductivity": {
        ("ecs", "arteries"): 9.1e-10 * mm / (Pa * s),
        ("ecs", "capillaries"): 1.0e-10 * mm / (Pa * s),
        ("ecs", "veins"): 2.0e-11 * mm / (Pa * s),
    },
    "surface_volume_ratio": {
        ("ecs", "arteries"): 3.0 / mm,
        ("ecs", "capillaries"): 9.0 / mm,
        ("ecs", "veins"): 3.0 / mm,
    },
    "flowrate": {
        "blood": 700 * mL / minute,
        "csf": 3.33 * mm**3 / s,
    },
    "pressure_drop": {
        ("arteries", "capillaries"): 50.0 * mmHg,
        ("capillaries", "veins"): 10.0 * mmHg,
        ("pvs_arteries", "pvs_capillaries"): 1.0 * mmHg,
        ("pvs_capillaries", "pvs_veins"): 0.25 * mmHg,
        ("c_prox", "c_dist"): 60.0 * mmHg,
    },
    "resistance": {
        "ecs": 4.56 * Pa * s / mm**3,
        "pvs_arteries": 1.14 * mmHg / (mL / minute),
        "pvs_capillaries": 32.4 * mmHg / (mL / minute),
        "pvs_veins": 1.75e-3 * mmHg / (mL / minute),
        ("ecs", "arteries"): 0.57 * mmHg / (mL / minute),
        ("ecs", "capillaries"): 125.31 * mmHg / (mL / minute),  # FIXME: Wrong value
        ("ecs", "veins"): 0.64 * mmHg / (mL / minute),
    },
    "diameter": {
        "arteries": 50.0 * um,
        "capillaries": 10.0 * um,
        "veins": 50.0 * um,
    },
    "solute_radius": {
        "inulin": 15.2e-7 * mm,  # Sauce? (1.52 nm?)
        "amyloid_beta": 0.9 * nm,
    },
    "membranes": {
        "layertype": {
            "glycocalyx": "fiber",
            "inner_endothelial_cleft": "pore",
            "endothelial_junction": "pore",
            "outer_endothelial_cleft": "pore",
            "basement_membrane": "fiber",
            "aef": "pore",
        },
        "thickness": {
            "glycocalyx": {
                "arteries": 400.0 * nm,
                "capillaries": 250.0 * nm,
                "veins": 100.0 * nm,
            },
            "inner_endothelial_cleft": 350.0 * nm,
            "endothelial_junction": 11.0 * mm,
            "outer_endothelial_cleft": 339.0 * nm,  # Total endothelial length 700nm
            "basement_membrane": {
                "arteries": 80.0 * nm,
                "capillaries": 30.0 * nm,
                "veins": 20.0 * nm,
            },
            "aef": 1000.0 * nm,
        },
        "elementary_radius": {
            "glycocalyx": 6.0 * nm,
            "inner_endothelial_cleft": 9.0 * nm,
            "endothelial_junction": {
                "arteries": 0.5 * nm,
                "capillaries": 2.5 * nm,
                "veins": 10.0 * nm,
            },
            "outer_endothelial_cleft": 9.0 * nm,
            "basement_membrane": {
                "arteries": 80.0 * nm,
                "capillaries": 30.0 * nm,
                "veins": 20.0 * nm,
            },
            "aef": {
                "arteries": 250.0 * nm,
                "capillaries": 10.0 * nm,
                "veins": 250.0 * nm,
            },
        },
        "fiber_volume_fraction": {"glycocalyx": 0.326, "basement_membrane": 0.5},
    },
}


def get_base_parameters():
    """Return a copy of the BASE_PARAMETERS, containing parameter values explicitly found in literature.
    Based on these values remaining paramters will be computed."""
    return {**BASE_PARAMETERS}


def unpack_shared_parameter(subdict: Dict, labels: List[str]):
    """Takes a dictionary, where the values are either Quantities/numbers or a nested dictionary, and
    returns a new dictionary where dictionaries-values are left untouched, but quantities are replaced by
    a dictionary with 'labels' as the keys, which all map to the initial value."
    """
    unpacked = {}
    for (
        layer,
        value,
    ) in subdict.items():  # Loop over the value given for each of the layers.
        unpacked[layer] = unpack_shared_quantity(value, labels)
    return unpacked


def unpack_shared_quantity(value, labels):
    """Given a value (Quantity or number) and a list of labels to share the parameters,
    returns a dictionary with the the labels as keys, and the value.

    If value is not a quantity or number, it is left untouched.
    """
    if isinstance(value, Quantity) or isinstance(
        value, numbers.Complex
    ):  # Single shared value for each compartment.
        return {vi: value for vi in labels}
    return value


def print_quantities(p, offset, depth=0):
    format_size = offset - depth * 2
    for key, value in p.items():
        if isinstance(value, dict):
            print(f"{depth*'  '}{str(key)}")
            print_quantities(value, offset, depth=depth + 1)
        else:
            print(f"{depth*'  '}{str(key):<{format_size+1}}: {value}")


def blood_resistance(p):
    Q = p["flowrate"]["blood"]
    dp = p["pressure_drop"]
    return {
        "arteries": dp[("arteries", "capillaries")] / Q,
        "capillaries": dp[("c_prox", "c_dist")] / Q,
        "veins": dp[("capillaries", "veins")] / Q,
    }


def get_permeabilities(p):
    R = {**p["resistance"], **blood_resistance(p)}
    k_e = p["permeability"]["ecs"]
    mu = {
        **unpack_shared_quantity(p["viscosity"]["csf"], SHARED_PARAMETERS["csf"]),
        **unpack_shared_quantity(p["viscosity"]["blood"], SHARED_PARAMETERS["blood"]),
    }
    length_area_ratio = R["ecs"] * k_e / mu["ecs"]
    k = {
        "ecs": k_e,
        **{
            key: length_area_ratio * mu[key] / R[key]
            for key in [*SHARED_PARAMETERS["pvs"], *SHARED_PARAMETERS["blood"]]
        },
    }
    return {key: val.to("mm^2") for key, val in k.items()}


def get_viscosities(params):
    viscosities = params["viscosity"]
    mu_blood = unpack_shared_quantity(viscosities["blood"], SHARED_PARAMETERS["blood"])
    mu_visc = unpack_shared_quantity(viscosities["csf"], SHARED_PARAMETERS["csf"])
    mu = {**mu_blood, **mu_visc}
    return mu


def get_porosities(params):
    phi = {**params["porosity"]}
    phi_B = params["vasculature_volume_fraction"]
    phi_PV = params["pvs_volume_fraction"]
    for vi, pvi in zip(SHARED_PARAMETERS["blood"], SHARED_PARAMETERS["pvs"]):
        fraction_vi = params["vasculature_fraction"][vi]
        phi[vi] = fraction_vi * phi_B
        phi[pvi] = fraction_vi * phi_PV
    return phi


def get_convective_fluid_transfer(params):
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
    return {key: val.to(1 / (Pa * s)) for key, val in T.items()}


def compute_partial_fluid_transfer(brain_volume, resistance, total_transfer):
    new_resistance = 1.0 / (total_transfer * brain_volume) - resistance
    return 1.0 / (new_resistance * brain_volume)

def compute_connected_fluid_transfer(brain_volume, flow_rate, pressure_drop):
    return flow_rate / (pressure_drop * brain_volume)


def get_osmotic_pressure(params, csf_factor=0.2):
    pi = unpack_shared_quantity(
        params["osmotic_pressure"]["blood"], SHARED_PARAMETERS["blood"]
    )
    pi_B = params["osmotic_pressure"]["blood"]
    for x in SHARED_PARAMETERS["all"]:
        if x in SHARED_PARAMETERS["blood"]:
            pi[x] = pi_B
        elif x in SHARED_PARAMETERS["csf"]:
            pi[x] = csf_factor * pi_B

    return pi


def get_osmotic_reflection(params):
    sigma = {}
    for interface in [*SHARED_PARAMETERS["bbb"], *SHARED_PARAMETERS["aef"]]:
        sigma[interface] = params["osmotic_reflection"]["all"]

    return sigma


def get_pressure_parameters(params):
    return {
        "permeability": get_permeabilities(params),
        "viscosity": get_viscosities(params),
        "porosity": get_porosities(params),
        "convective_fluid_transfer": get_convective_fluid_transfer(params),
        "osmotic_pressure": get_osmotic_pressure(params),
        "osmotic_reflection": get_osmotic_reflection(params),
    }


def make_dimless(params, param_units):
    dimless = {}
    for key, val in params.items():
        if isinstance(val, dict):
            dimless[key] = make_dimless(val, param_units[key])
        elif isinstance(val, Quantity):
            dimless[key] = val.to(param_units).magnitude
        else:
            dimless[key] = val
    return dimless


def get_interface_parameter(param, compartments):
    out = {}
    for i, j in combinations(compartments, 2):
        if (i, j) in param:
            out[(i, j)] = param[(i, j)]
            out[(j, i)] = param[(j, i)] if (j, i) in param else param[(i, j)]
        elif (j, i) in param:
            out[(j, i)] = param[(j, i)]
            out[(i, j)] = param[(i, j)] if (i, j) in param else param[(j, i)]
        else:
            out[(i, j)] = out[(j, i)] = 0.0
    return out
