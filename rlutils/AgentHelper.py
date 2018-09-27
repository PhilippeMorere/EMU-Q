# import numbers
import numpy as np


"""
AgentHelpers provide Agents with a series of function to access environment
state and action spaces, and Random Fourier Features.
"""


class AgentHelper:
    def __init__(self, glue, ftmap=None):
        if ftmap is None:
            self.nFeatures = glue.dS() + glue.dA()
        else:
            self.nFeatures = ftmap.nFeatures
        self.glue = glue
        self.ftmap = ftmap

    def toStateActionPair(self, ss, aa):
        ssaa = np.hstack([ss, aa])
        if self.ftmap is None:
            return ssaa
        else:
            return self.ftmap.toFeatures(ssaa)

    def isDiscA(self):
        return self.glue.env.discA

    def allDiscA(self):
        return np.linspace(0, 1, self.nA())

    def randDiscA(self, shape):
        nA = self.nA()
        return np.random.randint(0, nA, shape) / (nA-1)

    def sampleContA(self, nSamps):
        r = np.random.random((nSamps, self.dA()))
        low, high = self.glue.boundsA()
        return np.multiply(r, high - low) + low

    def randA(self, n):
        if self.isDiscA():
            return self.randDiscA(n)
        else:
            return self.sampleContA(n)

    def nA(self):
        return self.glue.nA()

    def boundsA(self):
        return self.glue.boundsA()

    def boundsS(self):
        return self.glue.boundsS()

    def dS(self):
        return self.glue.dS()

    def dA(self):
        return self.glue.dA()
