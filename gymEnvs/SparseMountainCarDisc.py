import numpy as np
import gym
from gym import spaces


class SparseMountainCarDisc(gym.Env):
    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        lowS = [-1.2, -0.07]
        highS = [0.6, 0.07]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        step = self.env.step(a)
        done = bool(step[0][0] >= self.env.env.goal_position)
        return step[0], 1.0 * done, done, {}
