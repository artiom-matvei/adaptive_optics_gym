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

from gym_AO.envs import AO_env_artiom
env = AO_env_artiom.AOEnvArtiom(wfs_mode=AO_env_artiom.PYRAMID_WFS, atmospheric_turbulence=False)
env.reset()
for i in range(1000):
    env.render() if i%10 else True
    env.step(0)
    print(i) if i%10 == 0 else True