import numpy as np
import nlopt


class AbsPolicy:
    def pick(self, s):
        raise NotImplementedError


class RandomPolicy(AbsPolicy):
    def __init__(self, agentHelper):
        self.agentHelper = agentHelper
        if self.agentHelper.isDiscA():
            self.dA = self.agentHelper
            self.pick = self.pickDisc
        else:
            self.pick = self.pickCont

    def pickDisc(self, s):
        return self.agentHelper.randDiscA((len(s), self.dA))

    def pickCont(self, s):
        return self.agentHelper.sampleContA(len(s))


class eGreedyPolicy(AbsPolicy):
    def __init__(self, agentHelper, scoreFn, epsilon=0.0):
        self.scoreFn = scoreFn
        self.agentHelper = agentHelper
        self.epsilon = epsilon

        self.dA = self.agentHelper.dA()

        if self.agentHelper.isDiscA():
            self.allAs = self.agentHelper.allDiscA().reshape(-1, 1)
            self.pick = self.pickDisc
        else:
            self.noSampsApprox = 15
            self.pick = self.pickCont

            # Setup nlopt
            self.currentState = None

            def __acq_fun_maximize(_x, grad):
                s = self.currentState.reshape(1, -1)
                a = _x.reshape(1, -1)
                score = float(self.scoreFn(s, a))
                return score

            opt_maxeval = 8
            self.opt = nlopt.opt(nlopt.LN_COBYLA, self.agentHelper.dA())
            boundsA = self.agentHelper.boundsA()
            self.opt.set_lower_bounds(boundsA[0])
            self.opt.set_upper_bounds(boundsA[1])
            self.opt.set_maxeval(opt_maxeval)
            self.opt.set_max_objective(__acq_fun_maximize)

    def pickDisc(self, s):
        randMask = np.random.random((len(s), )) < self.epsilon

        # Greedy
        repAs = np.vstack([self.allAs]*len(s))
        scores = self.scoreFn(np.repeat(s, len(self.allAs), axis=0),
                              repAs).reshape(len(s), -1)
        maxId = np.argmax(scores, axis=1)
        aa = self.allAs[maxId].reshape(-1, self.dA)

        # Combine greedy and random
        if np.any(randMask):
            randa = self.agentHelper.randDiscA((len(s), self.dA))
            return randa * randMask + aa * ~randMask
        return aa

    def pickCont2(self, s):
        aa = []
        for si in s:
            a = self.agentHelper.sampleContA(1)
            if np.random.random() >= self.epsilon:
                self.currentState = si
                a = a.reshape(-1)
                a = self.opt.optimize(a).reshape(1, -1)
            aa.append(a)
        return np.vstack(aa)

    def pickCont(self, s):
        randMask = np.random.random((len(s), )) < self.epsilon

        # Greedy
        sampsAs = self.agentHelper.sampleContA(self.noSampsApprox)
        repAs = np.vstack([sampsAs]*len(s))
        scores = self.scoreFn(np.repeat(s, len(sampsAs), axis=0),
                              repAs).reshape(len(s), -1)
        maxId = np.argmax(scores, axis=1)
        aa = sampsAs[maxId].reshape(-1, self.dA)

        # Combine greedy and random
        if np.any(randMask):
            randa = self.agentHelper.sampleContA(len(s))
            return randa * randMask + aa * ~randMask
        return aa
