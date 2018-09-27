import numpy as np
import gym
from gym import spaces


class LunarLander(gym.Env):
    def __init__(self):
        self.env = gym.make("LunarLanderContinuous-v2")
        lowS = [-1.2, -0.2, -3.3, -2.8, -3.4, -4, 0, 0]
        highS = [1.2, 1.6, 2, 0.8, 2.5, 9, 1.0, 1.0]
        lowA = [-1, -1]
        highA = [1, 1]
        self.action_space = spaces.Box(np.array(lowA), np.array(highA))
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        next_obs, _, done, __ = self.env.step(np.array(a).reshape(-1))

        if done:
            reward = -100
        elif abs(next_obs[0]) < 0.05 and abs(next_obs[1]) < 0.05:
            reward = 0
            done = True
        else:
            reward = -1

        return next_obs, reward, done, {}
