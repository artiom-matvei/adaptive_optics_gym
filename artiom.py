#%%
import datetime
import IPython
from hcipy import Wavefront, imshow_field, large_poisson
from matplotlib import animation, pyplot as plt
import numpy as np
from gym_AO.envs import AO_env_artiom

env = AO_env_artiom.AOEnvArtiom(wfs_mode=AO_env_artiom.PYRAMID_WFS, atmospheric_turbulence=True)
state, _ = env.reset(deformable_mirror_flat=False)
env.render()

leakage = 0.01
gain = 0.5
#%%
from matplotlib.animation import FuncAnimation

# Assuming your render() method returns a figure
def animate_ao_system(env, num_frames=50):
    # Get the initial figure from render
    fig = env.render()
    
    def update(frame):
        # Clear the current figure
        fig.clear()
        
        # Take an action (you might want to modify this)
        wf_dm = env.deformable_mirror.forward(env.telescope.wf_wfs)
        wf_pyr = env.wfs.forward(wf_dm)

        env.camera.integrate(wf_pyr, 1)
        wfs_image = large_poisson(env.camera.read_out()).astype(np.float64)
        wfs_image /= np.sum(wfs_image)

        diff_image = wfs_image-env.image_ref
        env.step(action=(1-leakage) * env.deformable_mirror.actuators - gain * env.reconstruction_matrix.dot(diff_image))
        
        # Get new figure content
        env.render()
        
        # Optional: adjust layout
        # fig.tight_layout()
    
    anim = FuncAnimation(fig, update, 
                        frames=num_frames,
                        interval=200,  # 50ms between frames
                        blit=False)
    return anim

# Usage:
env.reset(deformable_mirror_flat=False)

animation = animate_ao_system(env)

# Save the animation
animation.save(f'exps/anim-{datetime.datetime.now()}.mp4', writer='ffmpeg')
# %%
