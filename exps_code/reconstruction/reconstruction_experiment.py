def randomize_actuators(env):
    env.deformable_mirror.random(0.2 * env.telescope.wavelength_wfs)
    random_actuators = env.deformable_mirror.actuators
    return random_actuators
