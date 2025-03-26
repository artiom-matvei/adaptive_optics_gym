def get_reconstruction_matrix():
   # Create the interaction matrix
    probe_amp = 0.01 * wavelength_wfs
    slopes = []

    wf = Wavefront(magellan_aperture, wavelength_wfs)
    wf.total_power = 1

    for ind in range(num_modes):
        if ind % 10 == 0:
            print("Measure response to mode {:d} / {:d}".format(ind+1, num_modes))
        slope = 0

        # Probe the phase response
        for s in [1, -1]:
            amp = np.zeros((num_modes,))
            amp[ind] = s * probe_amp
            deformable_mirror.actuators = amp

            dm_wf = deformable_mirror.forward(wf)
            wfs_wf = pwfs.forward(dm_wf)

            camera.integrate(wfs_wf, 1)
            image = camera.read_out()
            image /= np.sum(image)

            slope += s * (image-image_ref)/(2 * probe_amp)

        slopes.append(slope)

    slopes = ModeBasis(slopes) 