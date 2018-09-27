import numpy as np

from rlutils.Normaliser import Normaliser

"""
This is the glue (wrapper) between RL environments from Env.py and the runner.
EnvGlue takes care of state and action space normalisation.
"""


class AbsEnvGlue:
    def __init__(self, env):
        self.env = env

    def resetEnv(self):
        raise NotImplementedError

    def stepEnv(self, a):
        raise NotImplementedError

    def nA(self):
        if not self.env.discA:
            raise ValueError("Environment action space is continuous.")
        if self.env.dA > 1:
            raise ValueError("Environment action space is multi dimensional.")
        return self.env.nA

    def boundsS(self):
        return np.array(self.env.lowS), np.array(self.env.highS)

    def boundsA(self):
        return np.array(self.env.lowA), np.array(self.env.highA)

    def dS(self):
        return self.env.dS

    def dA(self):
        return self.env.dA


class EnvGlue(AbsEnvGlue):
    def __init__(self, env, lowS=None, highS=None, normalised=True):
        super(EnvGlue, self).__init__(env)

        self.highA = [h - (1 if self.env.discA else 0) for h in self.env.highA]
        if lowS is None:
            lowS = self.env.lowS
        if highS is None:
            highS = self.env.highS
        self.normalised = normalised
        if normalised:
            self.nrms = Normaliser(lowS, highS, True)
            self.nrma = Normaliser(self.env.lowA, self.highA, True)

    def stepEnv(self, a):
        if self.normalised:
            rawA = self.nrma.unnormalise(a)
        else:
            rawA = a
        if self.env.discA:
            rawA = max(0, min(int(np.round(rawA)), self.highA[0]))
        rawS, r, done = self.env.step(rawA)
        rawS = rawS.reshape(-1)
        if self.normalised:
            return self.nrms.normalise(np.atleast_2d(rawS)), r, done
        else:
            return np.atleast_2d(rawS), r, done

    def stepsEnv(self, ss, aa):
        if self.normalised:
            rawAA = self.nrma.unnormalise(aa)
            rawSS = self.nrms.unnormalise(ss)
        else:
            rawAA = aa
            rawSS = ss
        if self.env.discA:
            rawAA = np.clip(np.round(rawAA), 0, self.highA[0])
        rawSSP, RR, DONE = self.env.steps(rawSS, rawAA)
        if self.normalised:
            return self.nrms.normalise(rawSSP), RR, DONE
        else:
            return rawSSP, RR, DONE

    def costEnv(self, ss):
        if self.normalised:
            rawSS = self.nrms.unnormalise(ss)
        else:
            rawSS = ss
        return self.env.cost(rawSS)

    def resetEnv(self):
        rawS = self.env.reset()
        if self.normalised:
            return self.nrms.normalise(np.atleast_2d(rawS))
        else:
            return np.atleast_2d(rawS)

    def boundsA(self):
        if self.normalised:
            return self.nrma.boundsNormalised()
        else:
            return super().boundsA()

    def boundsS(self):
        if self.normalised:
            return self.nrms.boundsNormalised()
        else:
            return super().boundsS()
