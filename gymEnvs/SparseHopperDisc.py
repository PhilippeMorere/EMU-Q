import numpy as np
import gym
from gym import spaces


class SparseHopperDisc(gym.Env):
    def __init__(self):
        self.env = gym.make("Hopper-v2")
        lowS = [-0.1, -0.25, -0.4, -0.4, -0.32] + [-7] * 6
        highS = [1.3, 0.02, 0.025, 0.03, 0.7] + [7] * 6
        self.action_space = spaces.Discrete(27)
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        a2 = np.array([(a // 9) - 1, ((a % 9) // 3) - 1, ((a % 9) % 3) - 1])
        step = self.env.step(a2)
        next_obs = step[0]
        posafter, height, ang = self.env.env.sim.data.qpos[0:3]

        if height > 1.3:
            reward = 1
            done = True
        elif abs(ang) > 0.2 or height < 0.7:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return next_obs, reward, done, {}
