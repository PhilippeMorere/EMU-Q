import numpy as np
import gym
from gym import spaces


class SparseDoublePendulumDisc(gym.Env):

    def __init__(self):
        self.env = gym.make("Acrobot-v1")
        lowS = [-1, -1, -42, -1, -1, -74]
        highS = [1,   1, 42,  1,  1,   74]

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        step = self.env.step(int(a))
        done = (step[1] == 0)
        reward = step[1] + 1
        return step[0], reward, done, {}
