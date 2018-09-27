import numpy as np
from tqdm import tqdm

from rlutils.ABLR import ABLR, fixedMeanABLR
from rlutils.Policies import eGreedyPolicy
from rlutils.Agents import AbsTDAgent


class Method(AbsTDAgent):
    def __init__(self, agtHlp, alphaQ=0.1, betaQ=0.1, alphaU=0.1, betaU=0.1,
                 kappa=1.0, gamma=0.99, maxVIiter=30, tm=None):
        self.agtHlp = agtHlp
        self.modelQ = ABLR(agtHlp.nFeatures, alphaQ, betaQ, computeSninv=False)
        sigma = agtHlp.ftmap.sigma
        self.modelU = fixedMeanABLR(agtHlp.nFeatures, alphaU, betaU, 0.0,
                                    sigma[0])
        self.kappa = kappa
        self.gamma = gamma
        self.tm = tm
        self.maxVIiter = maxVIiter
        self.epsLSTD = 0.1  # Stop LSTD loop if model change under this value
        self.greedyPolicy = eGreedyPolicy(agtHlp, self.score, 0.0)
        self.nSampsACont = 10
        self.varMax = 1.0 / alphaU
        self.kappa /= self.varMax
        print("Varmax={}, Kappa={}".format(self.varMax, self.kappa))

    def update(self, s, a, r, sp):
        """
        Adds one data point at a time.
        """
        ap = self.greedyPolicy.pick(sp)
        ftsa = self.agtHlp.toStateActionPair(s, a)
        ftsap = self.agtHlp.toStateActionPair(sp, ap)

        # Update Q
        qTarget = r + self.gamma * self.modelQ.predictMean(ftsap)
        self.modelQ.update(ftsa, qTarget)

        # Update U
        self._updateU(ftsa, ftsap, s, a, sp)

    def _updateU(self, ftsa, ftsap, s, a, sp):
        if self.agtHlp.isDiscA():
            allAs = self.agtHlp.allDiscA().reshape(-1, 1)
        else:
            allAs = self.agtHlp.sampleContA(self.nSampsACont)
        repSp = np.repeat(sp, len(allAs), axis=0)
        sampsFtsap = self.agtHlp.toStateActionPair(repSp, allAs)
        varQsp = np.mean(self.modelQ.predictVar(sampsFtsap))

        Usap = self.modelU.predictMean(ftsap)

        usp = varQsp - self.varMax
        self.modelU.update(ftsa, usp + self.gamma * Usap)

    def score(self, s, a):
        sa = self.agtHlp.toStateActionPair(s, a)
        m = self.modelQ.predictMean(sa)
        u = self.modelU.predictMean(sa)
        return m + self.kappa * u

    def endOfEpUpdate(self):
        if not self.tm:
            return
        print("End of episode update")
        # Retrieve all data
        data_s, data_a, data_r, data_sp = self.tm.getTransitions()

        # update for Q
        ftsa = self.agtHlp.toStateActionPair(data_s, data_a)
        # Init
        it = 0
        w = self.modelQ.mn
        prevW = w + 2 * self.epsLSTD
        pbar = tqdm(total=self.maxVIiter)
        while np.linalg.norm(prevW - w, 2) > self.epsLSTD and \
                it < self.maxVIiter:
            it += 1
            prevW = w

            data_ap = self.greedyPolicy.pick(data_sp)
            ftsap = self.agtHlp.toStateActionPair(data_sp, data_ap)
            qTargets = data_r + self.gamma * self.modelQ.predictMean(ftsap)
            self.modelQ.updateTargets(ftsa, qTargets)
            self.modelQ._recompute()

            w = self.modelQ.mn
            pbar.update(1)
        pbar.update(self.maxVIiter-it)
        pbar.close()
        print("")
        print("\tVI iterations for Q:{}".format(it))

        if self.kappa == 0:
            return
        # update for U
        data_ap = self.greedyPolicy.pick(data_sp)
        ftsap = self.agtHlp.toStateActionPair(data_sp, data_ap)
        lenS = data_s.shape[0]
        if self.agtHlp.isDiscA():
            allA = self.agtHlp.allDiscA().reshape(-1, 1)
        else:
            allA = self.agtHlp.sampleContA(self.nSampsACont)
        repAllA = np.vstack([allA] * lenS)
        repSp = np.repeat(data_sp, len(allA), 0)
        sampsFtsap = self.agtHlp.toStateActionPair(repSp, repAllA)
        allVarQsp = self.modelQ.predictVar(sampsFtsap).reshape(-1, lenS)
        varQ_sp = np.mean(allVarQsp, 0).reshape(-1, 1)
        # Init
        it = 0
        w = self.modelU.mn
        prevW = w + 2 * self.epsLSTD
        pbar = tqdm(total=self.maxVIiter)
        while np.linalg.norm(prevW - w, 2) > self.epsLSTD and \
                it < self.maxVIiter:
            it += 1
            prevW = w

            self._endEpUpdateU(ftsa, ftsap, data_s, data_a, data_sp, varQ_sp)
            self.modelU._recompute()

            w = self.modelU.mn
            pbar.update(1)
        pbar.update(self.maxVIiter-it)
        pbar.close()
        print("")
        print("\tVI iterations for U:{}".format(it))

    def _endEpUpdateU(self, ftsa, ftsap, data_s, data_a, data_sp, varQ_sp):
        rExpl = varQ_sp - self.varMax
        Usap = self.modelU.predictMean(ftsap)
        self.modelU.updateTargets(ftsa, rExpl + self.gamma * Usap)
