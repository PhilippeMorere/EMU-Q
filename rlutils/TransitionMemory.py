import numpy as np


class TransitionMemory:
    class EpisodeMemory:
        def __init__(self, noStepsMax, dS, dA, dR, onlyRewards=False):
            self.onlyRewards = onlyRewards
            if not onlyRewards:
                self.memS = np.zeros((noStepsMax, dS))
                self.memS2 = np.zeros((noStepsMax, dS))
                self.memA = np.zeros((noStepsMax, dA))
            self.memR = np.zeros((noStepsMax, dR))
            self.memId = 0

        def addStep(self, s, a, r, s2):
            if not self.onlyRewards:
                self.memS[self.memId] = s
                self.memS2[self.memId] = s2
                self.memA[self.memId] = a
            self.memR[self.memId] = r
            self.memId += 1

        def getS(self):
            if self.onlyRewards:
                raise NotImplementedError
            return self.memS[:self.memId, :]

        def getS2(self):
            if self.onlyRewards:
                raise NotImplementedError
            return self.memS2[:self.memId, :]

        def getA(self):
            if self.onlyRewards:
                raise NotImplementedError
            return self.memA[:self.memId, :]

        def getR(self):
            return self.memR[:self.memId, :]

    def __init__(self, noEpisiodes, noStepsMax, dS, dA, dR=1,
                 onlyRewards=False):
        """
        Memory storing all transitions (state, action, reward, new state) for
        all episodes.
        :param noEpisiodes: number of episodes
        :param noStepsMax: maximum number of steps per episode
        :param dS: state space dimension
        :param dA: action space dimension
        :param dR: reward space dimension (default=1)
        :param onlyRewards: Wheter to only record rewards (default=False).
        """
        self.memEps = [self.EpisodeMemory(noStepsMax, dS, dA, dR, onlyRewards)
                       for _ in range(noEpisiodes)]
        self.onlyRewards = onlyRewards
        self.memId = 0

    def addStep(self, s, a, r, s2):
        """
        Add step to current episode
        :param s: state
        :param a: action
        :param r: reward
        :param s2: new state
        """
        self.memEps[self.memId].addStep(s, a, r, s2)

    def endEpisode(self):
        """
        End current episode memory. Start new episode memory.
        """
        if self.memId < len(self.memEps):
            self.memId += 1

    def getEpisodeTransitions(self, noEp=-1):
        """
        Retreive transitions from given episode.
        :pram noEp: episode id (default=-1 last episode)
        :returns: matrixes of states, actopms, rewards, and new states
        """
        if noEp == -1:
            noEp = self.memId

        rr = self.memEps[noEp].getR()
        if self.onlyRewards:
            return rr

        ss = self.memEps[noEp].getS()
        ss2 = self.memEps[noEp].getS2()
        aa = self.memEps[noEp].getA()
        return ss, aa, rr, ss2

    def getTransitions(self):
        """
        Retreive transitions from all episodes (concatenated).
        :returns: matrixes of states, actopms, rewards, and new states
        """
        if self.memId == 0:
            rr = self.memEps[0].getR()
            if self.onlyRewards:
                return rr
            ss = self.memEps[0].getS()
            ss2 = self.memEps[0].getS2()
            aa = self.memEps[0].getA()
        else:
            rr = np.vstack([m.getR() for m in self.memEps[:self.memId]])
            if self.onlyRewards:
                return rr
            ss = np.vstack([m.getS() for m in self.memEps[:self.memId]])
            ss2 = np.vstack([m.getS2() for m in self.memEps[:self.memId]])
            aa = np.vstack([m.getA() for m in self.memEps[:self.memId]])
        return ss, aa, rr, ss2

    def getLastTransitions(self, nTransitions):
        maxEpId = min(len(self.memEps)-1, self.memId)
        # Count transitions
        nn = np.cumsum([self.memEps[i].memId for
                        i in range(maxEpId, -1, -1)])

        # Not enough transitions in memory
        if nn[-1] <= nTransitions:
            return self.getTransitions()

        idx = np.where(nn > nTransitions)[0][0] - 1
        memIdcs = range(maxEpId, idx-1, -1)

        if len(memIdcs) == 1:
            rr = self.memEps[memIdcs[0]].getR()
            if self.onlyRewards:
                return rr
            ss = self.memEps[memIdcs[0]].getS()
            ss2 = self.memEps[memIdcs[0]].getS2()
            aa = self.memEps[memIdcs[0]].getA()
        else:
            rr = np.vstack([self.memEps[i].getR() for i in memIdcs])
            if self.onlyRewards:
                return rr
            ss = np.vstack([self.memEps[i].getS() for i in memIdcs])
            ss2 = np.vstack([self.memEps[i].getS2() for i in memIdcs])
            aa = np.vstack([self.memEps[i].getA() for i in memIdcs])

        return ss[:nTransitions, :], aa[:nTransitions, :], \
            rr[:nTransitions, :], ss2[:nTransitions, :]

    def getEpisodeLengths(self):
        """
        Retreive episode lengths
        :returns: list of episode lengths
        """
        return [m.memId for m in self.memEps[:self.memId]]


if __name__ == "__main__":
    tm = TransitionMemory(3, 10, 2, 1)
    tm.addStep([0, 1], 0, -1, [0, 2])
    tm.addStep([0, 1], 0, -1, [0, 2])
    tm.addStep([0, 1], 0, -1, [0, 2])
    tm.addStep([0, 1], 0, -1, [0, 2])
    tm.addStep([0, 1], 0, -1, [0, 2])
    tm.endEpisode()
    tm.addStep([0, 2], 0, -1, [0, 2])
    tm.addStep([0, 2], 0, -1, [0, 2])
    tm.addStep([0, 2], 0, -1, [0, 2])
    tm.addStep([0, 2], 0, -1, [0, 2])
    tm.addStep([0, 2], 0, -1, [0, 2])
    tm.endEpisode()
    tm.addStep([0, 3], 0, -1, [0, 2])
    tm.addStep([0, 3], 0, -1, [0, 2])
    tm.endEpisode()
    s, a, r, s2 = tm.getTransitions()
    print("ALL:", s, a, r, s2)
    s, a, r, s2 = tm.getEpisodeTransitions()
    print("ONE:", s, a, r, s2)
