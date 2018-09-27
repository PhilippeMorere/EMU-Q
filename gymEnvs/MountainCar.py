import numpy as np
import gym
from gym import spaces


class MountainCar(gym.Env):
    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        lowS = [-1.2, -0.07]
        highS = [0.6, 0.07]
        lowA = [-1.0]
        highA = [1.0]
        self.action_space = spaces.Box(np.array(lowA), np.array(highA))
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        if np.abs(a) < 0.33:
            a2 = 1
        else:
            a2 = int(np.sign(a)) + 1
        step = self.env.step(a2)
        done = bool(step[0][0] >= self.env.env.goal_position)
        reward = 0.0 if done else -1.0
        return step[0], reward, done, {}
