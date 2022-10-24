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
        ("arteries", "pvs_arteries"),
        ("capillaries", "pvs_capillaries"),
        ("veins", "pvs_veins"),
    ],
    "aef": [("pvs_arteries", "ecs"), ("pvs_capillaries", "ecs"), ("pvs_veins", "ecs")],
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
        "csf": 7.0e-4,
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
        "blood": 700 * mm**3 / s,
        "csf": 3.33 * mm**3 / s,
    },
    "pressure_drop": {
        ("arteries", "capillaries"): 50.0 * mmHg,
        ("capillaries", "veins"): 10.0 * mmHg,
        ("pvs_arteries", "pvs_capillaries"): 1.0 * mmHg,
        ("pvs_capillaries", "pvs_veins"): 0.25 * mmHg,
    },
    "resistance": {
        "ecs": 4.56 * Pa * s / mm**2,
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
    ):  # Single shared value for each (blood) compartment.
        return {vi: value for vi in labels}
    return value


def print_quantities(p):
    format_size = max([len(str(key)) for key in p])
    for key, value in p.items():            
        print(f"{str(key):<{format_size+1}}: {value}")
