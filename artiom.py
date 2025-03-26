#%%
# import gymnasium as gym
# import numpy as np
# import gym_AO


    

# env = gym.make("AO-Artiom-v0", wfs_mode='shwfs', atmospheric_turbulence=False)

# env.reset()
# for i in range(1000):
#     env.render() if i%10 else True
#     _, reward, done, _, _ = env.step(1)
#     print(i) if i%10 == 0 else True
#     if i%10 == 0:
#         env.render()
#         print(i)
#         print(reward)

import IPython
from hcipy import Wavefront, imshow_field, large_poisson
from matplotlib import animation, pyplot as plt
import numpy as np
from gym_AO.envs import AO_env_artiom
env = AO_env_artiom.AOEnvArtiom(wfs_mode=AO_env_artiom.PYRAMID_WFS, atmospheric_turbulence=False)
state, _ = env.reset(deformable_mirror_flat=False)
env.render()

leakage = 0.01
gain = 0.5

def create_closed_loop_animation():

    wf_ref = Wavefront(env.telescope.VLT_aperture, env.telescope.wavelength_wfs)
    PSF = env.telescope.propagator.forward(wf_ref).power
    PSF /= PSF.max()

    fig = plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title('DM surface shape')
    im1 = imshow_field(env.deformable_mirror.surface, vmin=-1e-6, vmax=1e-6, cmap='bwr')

    plt.subplot(1,3,2)
    plt.title('Wavefront sensor output')
    im2 = imshow_field(env.image_ref, env.wfs.output_grid)

    plt.subplot(1,3,3)
    plt.title('Science image plane')
    im3 = imshow_field(np.log10(PSF), vmin=-5, vmax=0)

    plt.close(fig)

    def animate(t):
        wf_dm = env.deformable_mirror.forward(env.telescope.wf_wfs)
        wf_pyr = env.wfs.forward(wf_dm)

        env.camera.integrate(wf_pyr, 1)
        wfs_image = large_poisson(env.camera.read_out()).astype(np.float64)
        wfs_image /= np.sum(wfs_image)

        diff_image = wfs_image-env.image_ref
        env.deformable_mirror.actuators = (1-leakage) * env.deformable_mirror.actuators - gain * env.reconstruction_matrix.dot(diff_image)

        phase = env.telescope.VLT_aperture * env.deformable_mirror.surface
        phase -= np.mean(phase[env.telescope.VLT_aperture>0])

        psf = env.telescope.propagator.forward(wf_dm).power
        im1.set_data(*env.telescope.pupil_grid.separated_coords, (env.telescope.VLT_aperture * env.deformable_mirror.surface).shaped)
        im2.set_data(*env.wfs.output_grid.separated_coords, wfs_image.shaped)
        im3.set_data(*env.telescope.focal_grid.separated_coords, np.log10(psf.shaped/psf.max()))

        return [im1, im2, im3]

    num_time_steps=21
    time_steps = np.arange(num_time_steps)
    anim = animation.FuncAnimation(fig, animate, time_steps, interval=160, blit=True)
    return IPython.display.HTML(anim.to_jshtml(default_mode='loop'))

create_closed_loop_animation() # this needs to be changed to save to file as currently it is not working

reconstruction_matrix = env.reconstruction_matrix
print(reconstruction_matrix)
for i in range(1000):
    env.render() if i%10 == 0 else True
    _, reward, _, _, _ = env.step(0)
    print(f"step: {i}, reward: {reward}") if i%10 == 0 else True

