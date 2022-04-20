PARAMS = {
    "porosity_ecs": 0.18
    , 'diffusion_constant': 1.03e-4  # mm^2/s
    , "timestep": 60.  # s
    , "endtime": 3600. #3600. * 0.05  # s (=3600 s/h * h)
    , "decay": 0.01 / 60.  # 1% CSF clearance / minute
    , 'injection_center':  (-4., 2., 2.)
    , 'injection_spread': 1.  # mm (constant in bell curve)
    , 'cube_side_length': 2.
    , 'brain_volume': 2450.  # microliters=mm^3
    , 'csf_volume': 245.  # mircoliters=mm^3
}
