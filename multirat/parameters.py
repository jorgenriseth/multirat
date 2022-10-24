"""
All parameters should be stated in lowercase. 

- Global parameters may be stated straight forward.
- Compartment specific parameters should be stated on the form"
    "{parameter_name}-{compartment_name}(-extra_info)" 
- Interface specific parameters should be stated on the form
    "{parameter_name}-{compartment1}-{compartment2}(-extra_info)"

Parameters shared between multiple compartments/interfaces should be declared as such in
the SHARED_PARAMS dict, according to 
    "{shared_name}: ["comp1", "comp2"]", 
    "{interface_shared_name}: [("comp1", "comp2"), ("comp3", "comp4")]

A specific parameter given in the PARAMS list will override any shared parameters.
"""

PARAMS = {
    "volume-brain": 2450.0,  # microliters=mm^3
    "volume-csf": 245.0,  # mircoliters=mm^3
    "timestep": 60,  # in sec
    "endtime": 3600.0 * 6,  # s  (=3600 s/h * [x] h)
    "Initial_condition_concentration": "Gaussian",  # "Gaussian"
    "decay": 0.01 / 60.0,  # 1% CSF clearance / minute
    "injection_center": (4.0, 2.0, 3.0),
    "injection_spread": 1.0,  # mm (constant in bell curve)
    "cube_side_length": 2.5,
    # Compartment Specific Parameters
    "density-csf": 1e-3,  # g(mm^3)
    "porosity-ecs": 0.18,  # 0.24000
    "porosity-pvs_arteries": 0.00060,
    "porosity-pvs_capillaries": 0.00030,
    "porosity-pvs_veins": 0.00210,
    "porosity-arteries": 0.00658,
    "porosity-capillaries": 0.00329,
    "porosity-veins": 0.02303,
    "diffusion_constant": 1.03e-4,  # mm^2/s
    "diffusion_inulin_eff-all": 1.03e-4,  # mm^2/s
    "diffusion_inulin_free-all": 2.98e-4,  # mm^2/s
    "diffusion_ab_eff-all": 1.6236e-4,  # mm^2/s
    "diffusion_ab_free-all": 1.80e-4,  # mm^2/s
    "permeability-arteries": 1.0e-4,
    "permeability-veins": 1.0e-4,
    "permeability-capillaries": 1.0e-4,
    "permeability-pvs_arteries": 3.0e-11,
    "permeability-pvs_veins": 1.95e-8,
    "permeability-pvs_capillaries": 3.54e-13,
    "permeability-ecs": 1.3e-9,  # 1.4e-8
    "viscosity-blood": 2.67e-3,  # Pa.s
    "viscosity-csf": 8.9e-4,  # Pa.s
    "hydraulic_conductivity-bbb": 2.0e-8,  # mm.s^-1
    "hydraulic_conductivity-aef": 5.0e-7,  # mm.s^-1
    "osmotic_pressure-blood": 20 * 133.322368,  # Pa
    "osmotic_pressure-csf": 0.4 * 20 * 133.322368,  # Pa
    "osmotic_reflection_inulin-bbb": 0.8,
    "osmotic_reflection_inulin-aef": 0.8,
    "osmotic_reflection_ab-bbb": 0.8,
    "osmotic_reflection_ab-aef": 0.8,
    "inulin_drag-aef": 0.0,
    "inulin_drag-pvs": 0.0,
    "pressure_drop_arteries_capillaries": 40.0 * 133.322368,  # Pa
    "pressure_drop_capillaries_veins": 13.0 * 133.322368,  # Pa
    "blood_flow": 1.83,  # mL/s
    "surface_area_to_volume_ratio": 19.0,  # mm^-1
    "pressure_drop_pvs_arteries_capillaries": 1 * 133.322368,  # Pa
    "pressure_drop_pvs_capillaries_veins": 1 * 133.322368,  # Pa
    "average_periarterial_flow": 2.0e-4,  # mm^3/s
    "permeability_inulin-bbb": 0.0,  # Does not cross the BBB
    "permeability_inulin-aef": 5.9e-6,  # mm.s^-1  (aef = astrocytic endfeet)
    "permeability_ab-bbb": 7.5e-5,  # mm.s^-1      (ab = amyloid beta)
    "permeability_ab-aef": 7.5e-8  # mm.s^-1
    ### 1D resistance (translation to 3D is performed in the codes from the computed brain volume)
    # , "resistance_pvs" : 320.24*133.33/(1000.0)*60.0 # Pa.s/(mm^3)
    ,
    "resistance-arteries": 1.0e-4,
    "resistance-veins": 1.0e-4,
    "resistance-capillaries": 1.0e-4,
    "resistance-pvs_arteries": 1.14 * 133.33 / (1000.0) * 60.0,  # Pa.s/(mm^3)
    "resistance-pvs_veins": 1.95e-8,
    "resistance-pvs_capillaries": 1.44e-9,
    "resistance-ecs": 2e-11,  # 1.4e-8
}

SHARED_PARAMS = {
    "bbb": [
        ("arteries", "pvs_arteries"),
        ("capillaries", "pvs_capillaries"),
        ("veins", "pvs_veins")
    ],
    "aef": [
        ("pvs_arteries", "ecs"),
        ("pvs_capillaries", "ecs"),
        ("pvs_veins", "ecs")
    ],
    "csf": ["ecs", "pvs_arteries", "pvs_capillaries", "pvs_veins"],
    "blood": ["arteries", "capillaries", "veins"],
    "all": ["ecs", "pvs_arteries", "pvs_capillaries", "pvs_veins", "arteries", "capillaries", "veins"]
}

def get_default_parameters():
    return {**PARAMS}


def get_compartment_parameter(name, compartment, extension="", parameters=None):
    if parameters is None:
        parameters = get_default_parameters()

    parameter_name = f"{name}_{compartment}{extension}"
    return parameters[parameter_name]


def get_interface_parameter(
    name, comp1, comp2, symmetric, extension="", parameters=None
):
    if parameters is None:
        parameters = get_default_parameters()

    try:
        return parameters[f"{name}_{comp1}_{comp2}{extension}"]
    except KeyError as e:
        if symmetric:
            return parameters[f"{name}_{comp2}_{comp1}{extension}"]
        else:
            raise ValueError(e)


def retrieve_aliases(param):
    split = param.split("-")
    return [word for word in split[1:] if word in SHARED_PARAMS]

def unpack_shared_parameters(parameters):
    newparameters = {}
    for param, value in parameters.items():
        aliases = retrieve_aliases(param)
        for alias in aliases:
            if type(SHARED_PARAMS[alias][0]) == str:
                newnames = [param.replace(alias, key) for key in SHARED_PARAMS[alias]]
            else:
                newnames = [param.replace(alias, "-".join(key)) for key in SHARED_PARAMS[alias]]
            
            for name in newnames:
                newparameters[name] = value
        
        if len(aliases) == 0:
            newparameters[param] = value
        
    return newparameters