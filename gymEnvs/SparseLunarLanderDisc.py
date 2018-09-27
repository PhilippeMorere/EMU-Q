import numpy as np
import gym
from gym import spaces


class SparseLunarLanderDisc(gym.Env):
    def __init__(self):
        self.env = gym.make("LunarLanderContinuous-v2")
        lowS = [-1.2, -0.2, -3.3, -2.8, -3.4, -4, 0, 0]
        highS = [1.2, 1.6, 2, 0.8, 2.5, 9, 1.0, 1.0]
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        a2 = np.array([(a // 3) - 1, (a % 3) - 1])
        next_obs, _, done, __ = self.env.step(a2)
        if done:
            reward = -1
        elif abs(next_obs[0]) < 0.05 and abs(next_obs[1]) < 0.05:
            reward = 1
            done = True
        else:
            reward = 0

        return next_obs, reward, done, {}
