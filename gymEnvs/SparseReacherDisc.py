import numpy as np
import gym
from gym import spaces


class SparseReacherDisc(gym.Env):
    def __init__(self):
        self.env = gym.make("Reacher-v2")
        lowS = [-1, -1, -1, -0.17, -0.2, -0.2, -49, -5.6, -0.22, -0.2, 0]
        highS = [1, 1, 1, 1, 0.2, 0.2, 87, 33, 0.4, 0.35, 0.0001]
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        return self.env.reset()

    def step(self, a):
        a2 = np.array([(a // 3) - 1, (a % 3) - 1])
        step = self.env.step(a2)
        next_obs = step[0]
        vec = self.env.env.get_body_com("fingertip") - \
            self.env.env.get_body_com("target")
        dist = np.linalg.norm(vec)

        done = (dist <= 0.015)
        reward = 1 * (dist <= 0.015)
        return next_obs, reward, done, {}
