import numbers
from itertools import permutations

from numpy import exp, sqrt, pi
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


# Dictionary defining various subsets of the compartments and interfaces which
# either share a parameter, or use the same functions to compute parameters.
SHARED_PARAMETERS = {
    "all": [  # ## General Diffusive Permeabilities (for e.g. amyloid beta)
        "ecs",
        "pvs_arteries",
        "pvs_capillaries",
        "pvs_veins",
        "arteries",
        "capillaries",
        "veins",
    ],
    "pvs": ["pvs_arteries", "pvs_capillaries", "pvs_veins"],
    "csf": ["ecs", "pvs_arteries", "pvs_capillaries", "pvs_veins"],
    "blood": ["arteries", "capillaries", "veins"],
    "large_vessels": ["arteries", "veins"],
    "bbb": [
        ("pvs_arteries", "arteries"),
        ("pvs_capillaries", "capillaries"),
        ("pvs_veins", "veins"),
    ],
    "aef": [("ecs", "pvs_arteries"), ("ecs", "pvs_capillaries"), ("ecs", "pvs_veins")],
    "membranes": [
        ("pvs_arteries", "arteries"),
        ("pvs_capillaries", "capillaries"),
        ("pvs_veins", "veins"),
        ("ecs", "pvs_arteries"),
        ("ecs", "pvs_capillaries"),
        ("ecs", "pvs_veins"),
    ],
    "connected": [
        ("pvs_arteries", "pvs_capillaries"),
        ("pvs_capillaries", "pvs_veins"),
        ("arteries", "capillaries"),
        ("capillaries", "veins"),
    ],
    "connected_blood": [("arteries", "capillaries"), ("capillaries", "veins")],
    "connected_pvs": [
        ("pvs_arteries", "pvs_capillaries"),
        ("pvs_capillaries", "pvs_veins"),
    ],
}

# Dictionary containing parameters with values found in literature, or values for which
# we have just assumed some value. All other parameters should be derived from these.
BASE_PARAMETERS = {
    "brain_volume": 2450.0 * mm ** 3,
    "csf_volume_fraction": 0.12,  #
    "human_brain_volume": 1.0e6 * mm ** 3,
    "human_brain_surface_area": 1.750e2 * mm ** 2,
    "diffusion_coefficient_free": {
        "inulin": 2.98e-4 * mm ** 2 / s,
        "amyloid_beta": 1.8e-4 * mm ** 2 / s,
    },
    "pressure_boundary": {
        "arteries": 120.0 * mmHg,
        "veins": 7.0 * mmHg,
        "pvs_arteries": 4.74 * mmHg,
        "pvs_veins": 3.36 * mmHg,
        "ecs": 3.74 * mmHg,
    },
    "tortuosity": 1.7,
    "osmotic_pressure": {"blood": 20.0 * mmHg},
    "osmotic_pressure_fraction": {
        "csf": 0.2  # # Osm. press. computed as this constat * osmotic_pressure-blood
    },
    "osmotic_reflection": {"inulin": {"membranes": 0.2, "connected": 0.0}},
    "porosity": {"ecs": 0.14},
    "vasculature_volume_fraction": 0.0329,
    "vasculature_fraction": {"arteries": 0.2, "capillaries": 0.1, "veins": 0.7},
    "pvs_volume_fraction": 0.003,
    "viscosity": {"blood": 2.67e-3 * Pa * s, "csf": 7.0e-4 * Pa * s},
    "permeability": {"ecs": 2.0e-11 * mm ** 2},
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
        "blood": 2.4 * mL / minute,
        #         "blood": 700 * mL / minute, # human?
        "csf": 3.33 * mm ** 3 / s,
    },
    "pressure_drop": {
        ("arteries", "capillaries"): 50.0 * mmHg,
        ("capillaries", "veins"): 10.0 * mmHg,
        ("pvs_arteries", "pvs_capillaries"): 1.0 * mmHg,
        ("pvs_capillaries", "pvs_veins"): 0.25 * mmHg,
        ("c_prox", "c_dist"): 60.0 * mmHg,
    },
    "resistance": {
        "ecs": 4.56 * Pa * s / mm ** 3,
        "pvs_arteries": 1.14 * mmHg / (mL / minute),
        "pvs_capillaries": 32.4 * mmHg / (mL / minute),
        "pvs_veins": 1.75e-3 * mmHg / (mL / minute),
    },
    "resistance_interface": {
        ("ecs", "arteries"): 0.57 * mmHg / (mL / minute),
        ("ecs", "capillaries"): 125.31 * mmHg / (mL / minute),  # FIXME: Wrong value
        ("ecs", "veins"): 0.64 * mmHg / (mL / minute),
    },
    "diameter": {"arteries": 50.0 * um, "capillaries": 10.0 * um, "veins": 50.0 * um},
    "solute_radius": {"inulin": 15.2e-7 * mm, "amyloid_beta": 0.9 * nm},  # Sauce?
    # Related to permeability of BBB. Since this work is restricted to inulin, only AEF is of interest.
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

PARAMETER_UNITS = {
    "permeability": "mm**2",
    "viscosity": "Pa * s",
    "porosity": "",
    "convective_fluid_transfer": "1 / (Pa * h)",
    "osmotic_pressure": "Pa",
    "osmotic_reflection": "",
    "diffusive_solute_transfer": "1 / s",
    "convective_solute_transfer": "1 / (Pa * s)",
    "effective_diffusion_inulin": "mm**2 / s",
}


def get_base_parameters():
    """Return a copy of the BASE_PARAMETERS, containing parameter values explicitly found in literature.
    Based on these values remaining paramters will be computed."""
    return {**BASE_PARAMETERS}


def get_shared_parameters():
    """Return a copy of the SHARED_PARAMETERS, defining various subsets of compartments and interfaces."""
    return {**SHARED_PARAMETERS}


def pvs(v):
    return f"pvs_{v}"


def isquantity(x):
    return isinstance(x, Quantity) or isinstance(x, numbers.Complex)


def print_quantities(p, offset, depth=0):
    """Pretty printing of dictionaries filled with pint.Quantities"""
    format_size = offset - depth * 2
    for key, value in p.items():
        if isinstance(value, dict):
            print(f"{depth*'  '}{str(key)}")
            print_quantities(value, offset, depth=depth + 1)
        else:
            if isquantity(value):
                print(f"{depth*'  '}{str(key):<{format_size+1}}: {value:.3e}")
            else:
                print(f"{depth*'  '}{str(key):<{format_size+1}}: {value}")


def distribute_subset_parameters(base, subsets):
    """Take any parameter entry indexed by the name of some subset (e.g. 'blood'), 
    and create a new entry for each of the compartments/interfaces included in the
    given subset."""
    extended = {}
    for param_name, param_value in base.items():
        if not isinstance(param_value, dict):
            extended[param_name] = param_value
        else:
            param_dict = {**param_value}  # Copy old values

            # Check if any of the entries refer to a subset of compartments...
            for idx, val in param_value.items():
                if idx in subsets:
                    # ... and insert one entry for each of the compartment in the subset.
                    for compartment in subsets[idx]:
                        param_dict[compartment] = val
            extended[param_name] = param_dict
    return extended


def make_dimless(params, param_units):
    """Converts all quantities to the units specified by
    param_units, before converting it to a dimless number."""
    dimless = {}
    for key, val in params.items():
        if isinstance(val, dict):
            dimless[key] = make_dimless(val, param_units[key])
        elif isinstance(val, Quantity):
            dimless[key] = val.to(param_units).magnitude
        else:
            dimless[key] = val
    return dimless


def get_interface_parameter(param, compartments, is_symmetric=False):
    """Create a dictionary with entries for each permutation of 
    two compartments. Entries which are not present will either be
    set to zero, or to the mirrored value if the parameter is symmetric."""
    out = {}
    for i, j in permutations(compartments, 2):
        if (i, j) in param:
            out[(i, j)] = param[(i, j)]
        elif (j, i) in param and is_symmetric:
            out[(i, j)] = param[(j, i)]
        else:
            out[(i, j)] = out[(i, j)] = 0.0
    return out


def get_effective_diffusion(params, solute):
    Dfree = params["diffusion_coefficient_free"][solute]
    tortuosity = params["tortuosity"]
    return {key: Dfree / tortuosity ** 2 for key in SHARED_PARAMETERS["all"]}


def get_porosities(params):
    phi = {"ecs": params["porosity"]["ecs"]}
    phi_B = params["vasculature_volume_fraction"]
    phi_PV = params["pvs_volume_fraction"]
    for vi in SHARED_PARAMETERS["blood"]:
        fraction_vi = params["vasculature_fraction"][vi]
        phi[vi] = fraction_vi * phi_B
        phi[pvs(vi)] = fraction_vi * phi_PV
    return phi


def get_viscosities(params):
    viscosities = params["viscosity"]
    return viscosities


def get_resistances(params):
    R = {**params["resistance"], **blood_resistance(params)}
    return R


def blood_resistance(params):
    Q = params["flowrate"]["blood"]
    dp = params["pressure_drop"]
    return {
        "arteries": dp[("arteries", "capillaries")] / Q,
        "capillaries": dp[("c_prox", "c_dist")] / Q,
        "veins": dp[("capillaries", "veins")] / Q,
    }


def get_permeabilities(p):
    R = get_resistances(p)
    mu = p["viscosity"]
    k = {"ecs": p["permeability"]["ecs"]}
    length_area_ratio = R["ecs"] * k["ecs"] / mu["ecs"]
    for comp in [*SHARED_PARAMETERS["pvs"], *SHARED_PARAMETERS["blood"]]:
        k[comp] = length_area_ratio * mu[comp] / R[comp]

    return {key: val.to("mm^2") for key, val in k.items()}


def get_convective_fluid_transfer(params):
    T = {}
    V = params["human_brain_volume"]
    # Compute membrane transfer coefficients.
    # FIXME: Apparent error in resistance ecs-capillaries.
    for vi in SHARED_PARAMETERS["blood"]:
        L_ecs_vi = params["hydraulic_conductivity"][("ecs", vi)]
        surface_ratio = params["surface_volume_ratio"][("ecs", vi)]
        T_ecs_vi = L_ecs_vi * surface_ratio
        R_ecs_vi = params["resistance_interface"][("ecs", vi)]

        T[("ecs", pvs(vi))] = 1.0 / (V * R_ecs_vi)
        T[(pvs(vi), vi)] = compute_partial_fluid_transfer(V, R_ecs_vi, T_ecs_vi)

    # Compute connected transfer coefficients.
    for vi, vj in SHARED_PARAMETERS["connected_blood"]:
        Q = params["flowrate"]
        dp = params["pressure_drop"]
        T[(vi, vj)] = compute_connected_fluid_transfer(V, Q["blood"], dp[(vi, vj)])
        T[(pvs(vi), pvs(vj))] = compute_connected_fluid_transfer(
            V, Q["csf"], dp[(pvs(vi), pvs(vj))]
        )
    return {key: val.to(1 / (Pa * s)) for key, val in T.items()}


def compute_partial_fluid_transfer(brain_volume, resistance, total_transfer):
    new_resistance = 1.0 / (total_transfer * brain_volume) - resistance
    return 1.0 / (new_resistance * brain_volume)


def compute_connected_fluid_transfer(brain_volume, flow_rate, pressure_drop):
    return flow_rate / (pressure_drop * brain_volume)


def get_osmotic_pressure(params):
    pi_B = params["osmotic_pressure"]["blood"]
    csf_factor = params["osmotic_pressure_fraction"]["csf"]
    pi = {}
    for x in SHARED_PARAMETERS["all"]:
        if x in SHARED_PARAMETERS["blood"]:
            pi[x] = pi_B
        elif x in SHARED_PARAMETERS["csf"]:
            pi[x] = csf_factor * pi_B

    return pi


def get_osmotic_reflection(params, solute):
    sigma = {}
    for interface in SHARED_PARAMETERS["membranes"]:
        sigma[interface] = params["osmotic_reflection"][solute]["membranes"]

    for interface in SHARED_PARAMETERS["connected"]:
        # TODO: Verify correctness here as well.
        sigma[interface] = params["osmotic_reflection"][solute]["connected"]
    return sigma


def get_convective_solute_transfer(params, solute):
    sigma = get_osmotic_reflection(params, solute)
    G = get_convective_fluid_transfer(params)
    return {ij: G[ij] * (1 - sigma[ij]) for ij in G}


def diffusive_permeabilities_inulin(params):
    P = {}
    # Permeability over membranes.
    for vi in SHARED_PARAMETERS["blood"]:
        d_vi = params["diameter"][vi]
        R_aef_vi = diffusive_resistance_aef_inulin(params, vi)
        P[(pvs(vi), vi)] = 0.0 * mm / s
        P[("ecs", pvs(vi))] = 1.0 / (pi * d_vi * R_aef_vi)

    # Assume purely convection-driven transport between connected compartments.
    for i, j in SHARED_PARAMETERS["connected"]:
        P[(i, j)] = 0.0 * mm / s

    return {key: val.to("mm / s") for key, val in P.items()}


def diffusive_resistance_aef_inulin(params, vessel):
    D_free = params["diffusion_coefficient_free"]["inulin"]
    membranes = params["membranes"]

    thickness = membranes["thickness"]["aef"]
    B_aef = membranes["elementary_radius"]["aef"][vessel]
    solute_radius = params["solute_radius"]["inulin"]
    D_eff = diffusion_porous(D_free, solute_radius, B_aef)
    return resistance_aef_inulin(thickness, B_aef, D_eff)


def diffusion_porous(
    D_free: Quantity, solute_radius: Quantity, pore_radius: Quantity
) -> Quantity:
    beta = solute_radius / pore_radius
    return D_free * (
        1.0
        - 2.10444 * beta  # **6
        + 2.08877 * beta ** 3
        - 0.094813 * beta ** 5
        - 1.372 * beta ** 6
    )


def resistance_aef_inulin(layer_thickness, pore_radius, effective_diffusion):
    return layer_thickness / (2.0 * pore_radius * effective_diffusion)


def get_diffusive_solute_transfer_inulin(params):
    P = diffusive_permeabilities_inulin(params)
    surf_volume_ratio = params["surface_volume_ratio"]
    L = {}
    for vi in SHARED_PARAMETERS["blood"]:
        L[("ecs", pvs(vi))] = P[("ecs", pvs(vi))] * surf_volume_ratio[("ecs", vi)]
        L[(pvs(vi), vi)] = P[(pvs(vi), vi)] * surf_volume_ratio[("ecs", vi)]
    return {key: val.to(1 / (s)) for key, val in L.items()}


def get_boundary_hydraulic_permeabilities(p):
    # FIXME: This function is seemingly wrong.
    R_bdry = {
        "ecs": 2 * p["resistance"]["pvs_arteries"],
        "pvs_arteries": p["resistance"]["pvs_arteries"],
        "arteries": blood_resistance(p)["arteries"],
    }
    L_bdry = {}
    S = p["human_brain_surface_area"]
    for j in R_bdry:
        L_bdry[j] = 1.0 / (R_bdry[j] * S)
    return {key: value.to("mm / (Pa * s)") for key, value in L_bdry.items()}


def compute_parameters(params):
    return {
        "permeability": get_permeabilities(params),
        "viscosity": params["viscosity"],
        "porosity": get_porosities(params),
        "convective_fluid_transfer": get_convective_fluid_transfer(params),
        "osmotic_pressure": get_osmotic_pressure(params),
        "osmotic_reflection": get_osmotic_reflection(params, "inulin"),
        "effective_diffusion": get_effective_diffusion(params, "inulin"),
        "diffusive_solute_transfer": get_diffusive_solute_transfer_inulin(params),
        "convective_solute_transfer": get_convective_solute_transfer(params, "inulin"),
        "hydraulic_conductivity_bdry": get_boundary_hydraulic_permeabilities(params),
        "pressure_boundaries": params["pressure_boundary"],
    }


######################################
# GENERALIZED DIFFUSIVE PERMEABILITIES
######################################
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
        R_aef = Rvi["aef"]
        P[(pvs(vi), vi)] = (
            1.0 / (pi * dvi) / R_bbb if solute != "inulin" else 0.0 * mm / s
        )
        P[("ecs", pvs(vi))] = 1.0 / (pi * dvi) / R_aef

    # Assume purely convection-driven transport between connected compartments.
    for i, j in SHARED_PARAMETERS["connected"]:
        P[(i, j)] = 0.0 * mm / s

    return {key: val.to("mm / s") for key, val in P.items()}


def diffusive_resistance_membrane_layer(params, solute, vessel):
    membranes = distribute_membrane_params(params["membranes"])
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

        thickness = membranes["thickness"][layer][vessel]

        R[layer] = solute_resistance_layer(thickness, r, D_eff)
    return R


def distribute_membrane_params(membranes):
    """Take the membrane-parameter dictionary, and create a new dictionary with
    keys for each of the various vascular compartments, e.g.
    membranes[thickness][aef] -> membranes[thickness][aef][vi] for vi in blood."""
    unpacked = {}
    for param_name, param_values in membranes.items():
        # Do not separate layertype between different vessels
        if param_name == "layertype":
            unpacked[param_name] = param_values
            continue

        unpacked[param_name] = {}
        for layer, layer_value in param_values.items():
            if not isinstance(layer_value, dict):
                unpacked[param_name][layer] = {
                    vi: layer_value for vi in SHARED_PARAMETERS["blood"]
                }
            else:
                unpacked[param_name][layer] = {**layer_value}
    return unpacked


def diffusion_fibrous(D_free, solute_radius, fiber_radius, fiber_volume_fraction):
    return D_free * exp(
        -sqrt(fiber_volume_fraction) * (1.0 + solute_radius / fiber_radius)
    )


def solute_resistance_layer(layer_thickness, elementary_radius, effective_diffusion):
    return layer_thickness / (2.0 * elementary_radius * effective_diffusion)


def get_diffusive_solute_transfer(params, solute):
    P = diffusive_permeabilities(params, solute)
    surf_volume_ratio = params["surface_volume_ratio"]
    L = {}
    for vi in SHARED_PARAMETERS["blood"]:
        L[("ecs", f"pvs_{vi}")] = (
            P[("ecs", f"pvs_{vi}")] * surf_volume_ratio[("ecs", vi)]
        )
        L[(f"pvs_{vi}", vi)] = P[(f"pvs_{vi}", vi)] * surf_volume_ratio[("ecs", vi)]
    return {key: val.to(1 / (s)) for key, val in L.items()}
