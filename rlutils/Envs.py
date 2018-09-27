import gym
import math
import random
import numpy as np

"""
Envs provides a classic interface for all RL environments. It also makes it
easy to transform Gym environments to sparse reward problems.
"""


class AbsEnv:
    def __init__(self, lowS, highS, lowA, highA, discA=False):
        self.lowS, self.highS, self.lowA, self.highA = lowS, highS, lowA, highA
        self.dS, self.dA = len(lowS), len(lowA)
        self.discA = discA
        self.nA = int(highA[0]) if discA else -1

    def step(self, a):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class GymEnv(AbsEnv):
    def __init__(self, envName, render=False):
        self.env = gym.make(envName)

        # Get env bounds
        lowS = self.env.observation_space.low
        highS = self.env.observation_space.high
        # Discrete actions
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            discA = True
            lowA = [0]
            highA = [self.env.action_space.n]
        # Continuous actions
        else:
            discA = False
            lowA = self.env.action_space.low
            highA = self.env.action_space.high
        super(GymEnv, self).__init__(lowS, highS, lowA, highA, discA)

        self.render = render

    def step(self, a):
        if self.render:
            self.env.render()

        s, r, done, _ = self.env.step(a)
        return s, r, done

    def reset(self):
        return self.env.reset()


class SparseGymEnv(GymEnv):
    def checkSolved(self, s):
        raise NotImplementedError

    def __init__(self, envName, render=False):
        super(SparseGymEnv, self).__init__(envName, render)
        checkReward = None
        if envName == "MountainCar-v0":
            def checkReward(s):
                return bool(s[0] > 0.45)
        elif envName == "Acrobot-v1":
            def checkReward(s):
                ns = [math.atan2(s[1], s[0]), math.atan2(s[3], s[2])]
                return bool(-s[0] - np.cos(ns[0] + ns[1]) > 1.)
        elif envName == "Pendulum-v0":
            def checkReward(s):
                return s[0] >= 0.97

            def _checkDone(s, a, r):
                return False
            self.checkDone = _checkDone
        elif envName == "CartPole-v0":
            def checkReward(s):
                fallen = s[0] < -2.4 or s[0] > 2.4 \
                        or s[2] < -0.2094 or s[2] > 0.2094
                return not fallen

            def _checkDone(s, a, r):
                return r < 1
            self.checkDone = _checkDone

        elif envName == "Reacher-v2":
            def checkReward(s):
                vec = self.env.env.get_body_com("fingertip") - \
                    self.env.env.get_body_com("target")
                dist = np.linalg.norm(vec)
                return dist <= 0.015

            def _checkDone(s, a, r):
                return False
            self.checkDone = _checkDone

            def resetEnv():
                s = self.env.reset()
                print("d=", np.linalg.norm(self.env.env.goal))
                while np.linalg.norm(self.env.env.goal) > 0.18:
                    s = self.env.reset()
                return s
            self.reset = resetEnv
        else:
            raise ValueError("No solved criterion defined to create sparse",
                             "reward for environment {}".format(envName))
        self.checkSolved = checkReward

    def step(self, a):
        s, r, done = super(SparseGymEnv, self).step(a)
        r = self.checkSolved(s) * 1
        done = self.checkDone(s, a, r)
        return s, r, done

    def checkDone(self, s, a, r):
        return r == 1


class SparseOsbandChain(AbsEnv):
    def __init__(self, chainLen):
        self.chainLen = chainLen
        self.p = 1.0 - 1.0 / self.chainLen
        discA = True
        lowS, highS = [1], [chainLen]
        lowA, highA = [0], [2]
        super().__init__(lowS, highS, lowA, highA, discA)
        self.reset()

    def step(self, a):
        r = 0
        ss = self.s

        # Transitiion
        if a == 0:
            ss -= 1
        elif a == 1:
            if random.random() < self.p:
                ss += 1
            else:
                ss -= 1
        if ss > self.chainLen:
            ss = self.chainLen
        elif ss < 1:
            ss = 1

        # Reward
        if ss == self.chainLen:
            r = 1

        self.s = ss
        return np.array([self.s]), r, r == 1

    def reset(self):
        self.s = 1
        return np.array([self.s])


class SemiSparseOsbandChain(SparseOsbandChain):
    def __init__(self, chainLen, rewardSparsity):
        """
        param rewardSparsity: indicates how sparse the domain is (0 to 1)
        """
        self.nRwds = int((1 - rewardSparsity) * (chainLen - 1))
        self.rewardIdx = None
        super().__init__(chainLen)

    def step(self, a):
        s, r, done = super().step(a)
        if s in self.rewardIdx:
            r = -1
        return s, r, done

    def reset(self):
        idx = np.arange(1, self.chainLen)
        np.random.shuffle(idx)
        self.rewardIdx = idx[0:self.nRwds]
        return super().reset()


class SparseExplorationChain(AbsEnv):
    """
    Typical problem to test exploration. This chain of specified length is
    hard to explore. Action 1 goes to the next state (increasing). Action 0
    always goes to state 1. A reward of 1 is given for the right-most state,
    otherwise 0 reward is given. Problem starts in state 1.
    """
    def __init__(self, chainLen):
        self.chainLen = chainLen
        discA = True
        lowS, highS = [1], [chainLen]
        lowA, highA = [0], [2]
        super().__init__(lowS, highS, lowA, highA, discA)
        self.reset()

    def step(self, a):
        r = 0

        # Transition
        if a == 0:
            self.s = 1
        elif a == 1 and self.s < self.chainLen:
            self.s += 1

        # Reward
        if self.s == self.chainLen and a == 1:
            r = 1

        return np.array([self.s]), r, r == 1

    def reset(self):
        self.s = 1
        return np.array([self.s])


class SemiSparseExplorationChain(SparseExplorationChain):
    def __init__(self, chainLen, rewardSparsity):
        """
        param rewardSparsity: indicates how sparse the domain is (0 to 1)
        """
        self.nRwds = int((1 - rewardSparsity) * (chainLen - 1))
        self.rewardIdx = None
        super().__init__(chainLen)

    def step(self, a):
        s, r, done = super().step(a)
        if s in self.rewardIdx:
            r = -1
        return s, r, done

    def reset(self):
        idx = np.arange(1, self.chainLen)
        np.random.shuffle(idx)
        self.rewardIdx = idx[0:self.nRwds]
        return super().reset()
