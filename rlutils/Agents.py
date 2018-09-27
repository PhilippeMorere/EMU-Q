from rlutils.LR_SGD import LR_SGD
from rlutils.Policies import eGreedyPolicy

"""
Provides an implementation of QLearning.
"""


class AbsAgent:
    def update(self, s, a, r, sp):
        raise NotImplementedError

    def endOfEpUpdate(self):
        raise NotImplementedError


class AbsTDAgent(AbsAgent):
    def score(self, s, a):
        raise NotImplementedError


class QLearning(AbsTDAgent):
    def __init__(self, agtHlp, gamma=0.99, **kwargs):
        """
        Parameters
        :param agtHlp: agentHelper object.
        :param learningRate: stochasticOptimisers object (optional.
        :param gamma: discount factor (optional).
        """
        self.agtHlp = agtHlp
        self.gamma = gamma
        self.greedyPolicy = eGreedyPolicy(agtHlp, self.score, 0.0)
        self.model = LR_SGD(M=agtHlp.nFeatures, **kwargs)

    def update(self, s, a, r, sp):
        """
        Adds one data point at a time.
        """
        ftsa = self.agtHlp.toStateActionPair(s, a)
        ap = self.greedyPolicy.pick(sp)
        sap = self.agtHlp.toStateActionPair(sp, ap)
        self.model.update(ftsa, r + self.gamma * self.model.predictMean(sap))

    def score(self, s, a):
        sa = self.agtHlp.toStateActionPair(s, a)
        return self.model.predictMean(sa)

    def endOfEpUpdate(self):
        pass
