import numpy as np
import gym
from gym import spaces


class DoublePendulum(gym.Env):
    def __init__(self):
        self.env = gym.make("Acrobot-v1")
        lowS = [-1, -1, -42, -1, -1, -74]
        highS = [1,   1, 42,  1,  1,   74]
        lowA = [-50]
        highA = [50]

        self.action_space = spaces.Box(np.array(lowA), np.array(highA))
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        step = self.env.step(1 * (int(np.sign(a)) > 0))
        return step[0], step[1], step[2], {}
