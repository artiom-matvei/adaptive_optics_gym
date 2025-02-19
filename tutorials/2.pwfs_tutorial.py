#%%
from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

# These modules are used for animating some of the graphs in our notebook.
from matplotlib import animation, rc
from IPython.display import HTML

# %%
wavelength_wfs = 842.0E-9
telescope_diameter = 6.5
zero_magnitude_flux = 3.9E10
stellar_magnitude = 0
# %%
num_pupil_pixels = 60
pupil_grid_diameter = 60/56 * telescope_diameter
pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

pwfs_grid = make_pupil_grid(120, 2 * pupil_grid_diameter)

magellan_aperture = evaluate_supersampled(make_magellan_aperture(), pupil_grid, 6)

imshow_field(magellan_aperture)
plt.xlabel('x position(m)')
plt.ylabel('y position(m)')
plt.colorbar()
plt.show()
# %%
num_actuators_across_pupil = 10
actuator_spacing = telescope_diameter / num_actuators_across_pupil
influence_functions = make_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing)
deformable_mirror = DeformableMirror(influence_functions)
num_modes = deformable_mirror.num_actuators
# %%
pwfs = PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=pupil_grid_diameter, pupil_diameter=telescope_diameter, wavelength_0=wavelength_wfs, q=3)
camera = NoiselessDetector(pwfs_grid)
# %%
wf = Wavefront(magellan_aperture, wavelength_wfs)
wf.total_power = 1

camera.integrate(pwfs.forward(wf), 1)

image_ref = camera.read_out()
image_ref /= image_ref.sum()

imshow_field(image_ref)
plt.colorbar()
plt.show()
# %%

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

        slope += s * (image - image_ref)/(2 * probe_amp)
    
    slopes.append(slope)

slopes = ModeBasis(slopes)
# %%
# What are we doing exactly?
# We are measuring the response in the WFS caused by an action on an actuator (a).
# We compose them together to create the interaction matrix A.
# The multiplication of A and a yields the wavefront sensor response:
# S = A * a

# Note that we want a certain WFS measurement S'. 
# To recover it, we can invert the interaction matrix A as below by using regularization techniques ensuring the invertability.
# a = (A^T A)^{-1} A^T S

rcond = 1E-3
reconstruction_matrix = inverse_tikhonov(slopes.transformation_matrix, rcond=rcond, svd=None)

# %%
print(reconstruction_matrix.shape,
slopes.transformation_matrix.shape)

# %%
spatial_resolution = wavelength_wfs / telescope_diameter
focal_grid = make_focal_grid(q=8, num_airy=20, spatial_resolution=spatial_resolution)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
norm = prop(wf).power.max()
# %%
deformable_mirror.random(0.2 * wavelength_wfs)

# %%
delta_t = 1E-3
leakage = 0.0
gain = 0.5
# %%
PSF_in = prop(deformable_mirror.forward(wf)).power

imshow_psf(PSF_in / norm, vmax=1, vmin=1e-5, spatial_resolution=spatial_resolution)
plt.show()
#%%
plt.plot(deformable_mirror.actuators)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Deformable Mirror Actuators')
plt.show()
# %%
def create_closed_loop_animation():

    PSF = prop(deformable_mirror(wf)).power

    fig = plt.figure(figsize=(14,3))
    plt.subplot(1,3,1)
    plt.title(r'DM surface shape ($\mathrm{\mu}$m)')
    im1 = imshow_field(deformable_mirror.surface/(1e-6), vmin=-1, vmax=1, cmap='bwr')
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title('Wavefront sensor output')
    im2 = imshow_field(image_ref, pwfs_grid)
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title('Science image plane')
    im3 = imshow_field(np.log10(PSF / norm), vmax=0, vmin=-5, cmap='inferno')
    plt.colorbar()

    plt.close(fig)

    def animate(t):
        wf_dm = deformable_mirror.forward(wf)
        wf_pyr = pwfs.forward(wf_dm)

        camera.integrate(wf_pyr, 1)
        wfs_image = camera.read_out().astype('float')
        wfs_image /= np.sum(wfs_image)

        diff_image = wfs_image - image_ref
        deformable_mirror.actuators = (1-leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(diff_image)
        
        phase = magellan_aperture * deformable_mirror.surface
        phase -= np.mean(phase[magellan_aperture>0])

        psf = prop(deformable_mirror(wf)).power

        im1.set_data(*pupil_grid.separated_coords, (magellan_aperture * deformable_mirror.surface).shaped / 1e-6)
        im2.set_data(*pwfs_grid.separated_coords, wfs_image.shaped)
        im3.set_data(*focal_grid.separated_coords, np.log10(psf.shaped / norm))

        return [im1, im2, im3]

    num_time_steps = 21 
    time_steps = np.arange(num_time_steps)
    anim = animation.FuncAnimation(fig, animate, time_steps, interval=160, blit=True)
    return HTML(anim.to_jshtml(default_mode='loop'))

create_closed_loop_animation()
# %%
