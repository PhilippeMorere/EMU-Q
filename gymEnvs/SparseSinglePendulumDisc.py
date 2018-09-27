import numpy as np
import gym
from gym import spaces


class SparseSinglePendulumDisc(gym.Env):
    def __init__(self):
        self.env = gym.make("Pendulum-v0")
        lowS = [-1, -1, -8]
        highS = [1, 1, 8]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        s = self.env.env.state
        a2 = (a - 2) * 2
        step = self.env.step(np.array([a2]))
        done = (s[0] > 0.95 and abs(s[1]) < 0.05)
        return step[0], 1.0 * done, done, {}
