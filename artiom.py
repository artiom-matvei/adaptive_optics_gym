# %%
import datetime
import itertools
import IPython
from hcipy import Wavefront, imshow_field, large_poisson
from matplotlib import animation, pyplot as plt
import numpy as np
from exps_code.reconstruction.reconstruction_experiment import randomize_actuators
from gym_AO.envs import AO_env_artiom
from matplotlib.animation import FuncAnimation

env = AO_env_artiom.AOEnvArtiom(
    wfs_mode=AO_env_artiom.PYRAMID_WFS, atmospheric_turbulence=False
)
state, _ = env.reset(deformable_mirror_flat=False)


INITIAL_ACTUATORS = randomize_actuators(env)

leakage_list = [0.1]#, 0.01] # biases actions towards zero, is it necessary?
gain_list = [0.25]#, 0.5, 1]
reconstruction_method = True
#%%
for leakage, gain in itertools.product(leakage_list, gain_list):

    def animate_ao_system(env, num_frames=50):
        fig = env.render()

        def update(frame):
            fig.clear()

            wf_dm = env.deformable_mirror.forward(env.telescope.wf_wfs)
            wf_pyr = env.wfs.forward(wf_dm)

            env.camera.integrate(wf_pyr, 1)
            wfs_image = large_poisson(env.camera.read_out()).astype(np.float64)
            wfs_image /= np.sum(wfs_image)

            diff_image = wfs_image - env.image_ref
            action = (
                (1 - leakage) * env.deformable_mirror.actuators
                - gain * env.reconstruction_matrix.dot(diff_image)
                if reconstruction_method
                else np.zeros_like(env.deformable_mirror.actuators)
            )
            env.step(action=action)

            env.render()

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=200,  # 50ms between frames
            blit=False,
        )
        return anim

    # Usage:
    env.reset()
    env.deformable_mirror.actuators = INITIAL_ACTUATORS

    animation = animate_ao_system(env)

    # Save the animation
    animation.save(
        f"exps/reconstruction/no_turb/anim-{gain}-{leakage}-{datetime.datetime.now()}.mp4",
        writer="ffmpeg",
    )
# %%
