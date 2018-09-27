import numpy as np
import gym
from gym import spaces


class SparseCartpoleSwingup(gym.Env):

    def __init__(self, render=False):
        self.env = gym.make("CartPole-v1")
        lowS = [-3.4, -8.0, -5.4, -8.7]
        highS = [3.4, 8.3, 6.35, 8.1]
        lowA = [-10]
        highA = [10]
        self.max_cart_pos = 3
        self.action_space = spaces.Box(np.array(lowA), np.array(highA))
        self.observation_space = spaces.Box(np.array(lowS), np.array(highS))

    def reset(self):
        s = self.env.reset()
        s2 = np.array([s[0], s[1], s[2] + np.pi, s[3]])
        self.env.env.state = s2
        return np.array(s2)

    def step(self, a):
        step = self.env.step(1 * (int(np.sign(a)) > 0))
        ss = step[0]

        if abs(ss[0]) > self.max_cart_pos:
            reward = -100
            done = True
        elif np.cos(ss[2]) > 0.8:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.env.env.steps_beyond_done = None

        return ss, reward, done, {}
