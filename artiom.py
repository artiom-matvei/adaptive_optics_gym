import gymnasium as gym
import numpy as np
import gym_AO


    

env = gym.make("AO-Artiom-v0", render_mode='atmosphere')

env.reset()
for i in range(1000):
    env.render() if i%10 else True
    env.step(1)
    print(i) if i%10 == 0 else True


