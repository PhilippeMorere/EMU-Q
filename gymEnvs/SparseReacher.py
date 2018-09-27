import numpy as np
import gym
from gym import spaces


class SparseReacher(gym.Env):
    def __init__(self):
        self.env = gym.make("Reacher-v2")
        lowS = [-1, -1, -1, -0.17, -0.2, -0.2, -49, -5.6, -0.22, -0.2, 0]
        highS = [1, 1, 1, 1, 0.2, 0.2, 87, 33, 0.4, 0.35, 0.0001]
        lowA = [-1, 1]
        highA = [1, 1]
        self.action_space = spaces.Box(np.array(lowA), np.array(highA))
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        step = self.env.step(np.array(a))
        next_obs = step[0]
        vec = self.env.env.get_body_com("fingertip") - \
            self.env.env.get_body_com("target")
        dist = np.linalg.norm(vec)

        done = (dist <= 0.015)
        reward = 1 * (dist <= 0.015)
        return next_obs, reward, done, {}
