from pint import Quantity
# from multirat.parameters.base import get_pressure_parameters


# def test_get_solute_parameters():
#     blood = ["arteries", "capillaries","veins"]
#     pvs = ["pvs-arteries", "pvs-capillaries", "pvs-veins"]
#     csf = ["ecs", *pvs]
#     compartments = [*csf, *blood]
#     params = get_solute_parameters()
#     for comp in csf:
#         for x in params:
#             assert comp in params[x]
#             assert not isinstance(params[x][comp], Quantity)


# def test_get_pressure_parameters():
#     params = get_pressure_parameters()
#     blood = ["arteries", "capillaries","veins"]
#     pvs = ["pvs-arteries", "pvs-capillaries", "pvs-veins"]
#     csf = ["ecs", *pvs]
#     compartments = [*csf, *blood]
#     params = get_pressure_parameters(compartments)
#     for x in params:
#         for comp in compartments:
#             assert comp in params[x]
#             assert not isinstance(params[x][comp], Quantity)

