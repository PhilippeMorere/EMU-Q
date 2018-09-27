import numpy as np
import uuid
import json
import os
import argparse

from rlutils.TransitionMemory import TransitionMemory
from rlutils.Runners import EnvRunner
from rlutils.EnvGlue import EnvGlue
from rlutils.Envs import SparseGymEnv, GymEnv
from rlutils.AgentHelper import AgentHelper
from Method import Method
from rlutils.Agents import QLearning
from rlutils.LR_SGD import ConstantRate
from rlutils.Policies import eGreedyPolicy

from features.RandomFourierFeatures import RFF
from features.FourierBasisFeatures import FourierBasisFeatures as FBF


def fix_spaces(env):
    """
    Set arbitrary space bounds for states variables missing bounds (inf)
    """
    l, h = env.lowS, env.highS
    for i in range(len(l)):
        if l[i] < -100:
            l[i] = 100
        if h[i] > 100:
            h[i] = 100
    return l, h


def gen_env(opt):
    nStep = opt.nStep
    nEp = opt.nEp
    nRFF = opt.nRFF
    kernelType = opt.kernelType
    if opt.gymEnv is not None:
        env = GymEnv(opt.gymEnv, render=opt.render)
        envName = opt.gymEnv
    elif opt.sparseGymEnv is not None:
        env = SparseGymEnv(opt.sparseGymEnv, render=opt.render)
        envName = opt.sparseGymEnv
    else:
        raise ValueError("An environment needs to be specified")

    lowS, highS = fix_spaces(env)
    glue = EnvGlue(env, lowS, highS)
    sigma = np.array([opt.sigmaS] * glue.dS() + [opt.sigmaA] * glue.dA())
    alphaQ, betaQ, alphaU, betaU = opt.alphaQ, opt.betaQ, opt.alphaU, opt.betaU

    return env, envName, nEp, nStep, nRFF, kernelType, lowS, highS, glue, sigma, alphaQ, betaQ, alphaU, betaU


def main(opt):
    # Load default values and env
    env, envName, nEp, nStep, nRFF, kernelType, lowS, highS, glue, sigma, alphaQ, betaQ, alphaU, betaU = gen_env(opt)

    print(opt)
    # Generate directory for experiment
    if opt.enableLogFile:
        expName, i = "{}_{}".format(opt.expName, envName), 1
        while os.path.exists("exps/{}{}".format(expName, i)):
            i += 1
        dirName = "exps/{}{}".format(expName, i)
        os.makedirs(dirName)

    # Run experiment
    allRets = []
    for repI in range(opt.nRepeat):
        tm = TransitionMemory(nEp, nStep, glue.dS(), glue.dA())

        # Agt, Pol
        if opt.featureType == "RFF":  # Random Fourier Features
            ftmap = RFF(nRFF, glue.dS() + glue.dA(), sigma, kernelType)
        elif opt.featureType == "FBF":  # Fourier basis features
            ftmap = FBF(glue.dS() + glue.dA(), opt.fourierBasisOrder)
        agtHlp = AgentHelper(glue, ftmap)

        if opt.agent == "method":
            agent = Method(agtHlp, gamma=opt.gamma, alphaQ=alphaQ, betaQ=betaQ,
                           alphaU=alphaU, betaU=betaU, tm=tm,
                           maxVIiter=opt.maxVIiterQ, kappa=opt.kappa)
        elif opt.agent == "QLearning":
            lr = ConstantRate(opt.learningRate)
            agent = QLearning(agtHlp, gamma=opt.gamma, learningRate=lr)
        else:
            raise ValueError("Unknown agent {}".format(opt.agent))
        policy = eGreedyPolicy(agtHlp, agent.score, opt.epsilon)

        # Runner
        runner = EnvRunner(glue, policy, agent, tm, opt.verbose)

        # Go
        runner.run(nEp, nStep, opt.stopWhenSolved)

        # Keep track of returns
        idx = np.cumsum([0] + tm.getEpisodeLengths())
        ss, _, rr, __ = tm.getTransitions()
        sMin, sMax = np.min(ss, 0), np.max(ss, 0)
        print("State min/max:\n{}\n{}".format(sMin, sMax))
        rets = [np.sum(rr[idx[i-1]:idx[i]]) for i in range(1, len(idx))]
        print("Repeat", repI, "finished.")
        print("Returns:\n", rets)
        allRets.append(rets)

        # Parse all variables to file
        del rr, _, ss, __
        if opt.enableLogFile:
            filename = "{}/vars_{}.json".format(dirName, uuid.uuid4().hex)
            with open(filename, 'w') as f:
                json.dump({k: repr(v) for k, v in vars().items()}, f,
                          indent=4, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", help="method, QLearning",
                        default="method")
    parser.add_argument("--gymEnv", help="Gym Enviornment to run",
                        default=None, type=str)
    parser.add_argument("--sparseGymEnv", help="Sparse Gym Enviornment to run",
                        default=None, type=str)
    parser.add_argument("--nStep", help="Number of steps per episode",
                        default=500, type=int)
    parser.add_argument("--nEp", help="Number of episodes", default=20,
                        type=int)
    parser.add_argument("--nRepeat", help="number of repetitions", default=1,
                        type=int)
    parser.add_argument("--stopWhenSolved", help="Stop repeat after goal is "
                        "reahced for the first time.",
                        action="store_true", default=False)
    parser.add_argument("--gamma", help="Reward discount value", default=0.99,
                        type=float)

    # Feature parameters
    parser.add_argument("--featureType", help="Type of features to use: "
                        "RFF for Random Fourier Features, "
                        "FBF for Fourier Basis Features.",
                        default="RFF", type=str)
    parser.add_argument("--nRFF", help="Number of RFF features", default=300,
                        type=int)
    parser.add_argument("--kernelType", help="RFF kernel type",
                        default="RBF", type=str)
    parser.add_argument("--sigmaS", help="State RFF features lengthscale",
                        default=0.35, type=float)
    parser.add_argument("--sigmaA", help="Action RFF features lengthscale",
                        default=1.0, type=float)
    parser.add_argument("--fourierBasisOrder", type=int,
                        help="Fourier basis feature order", default=3)

    # Algorithm parameters
    parser.add_argument("--kappa", help="Exploration-exploitation balance",
                        default=1.0, type=float)
    parser.add_argument("--epsilon", help="epsilon-greedy policy parameter",
                        default=0.0, type=float)
    parser.add_argument("--alphaQ", help="BLR weight prior precision for Q",
                        default=0.1, type=float)
    parser.add_argument("--betaQ", help="BLR noise precision for Q",
                        default=1.0, type=float)
    parser.add_argument("--alphaU", help="BLR weight prior precision for U",
                        default=0.1, type=float)
    parser.add_argument("--betaU", help="BLR noise precision for U",
                        default=1.0, type=float)
    parser.add_argument("--maxVIiterQ", help="Maximum number of VI iterations"
                        " at the end of each episode for Q", default=30,
                        type=int)
    parser.add_argument("--learningRate", help="QLearning learning rate",
                        default=0.5, type=float)

    # Logging
    parser.add_argument("--expName", help="Experiment name", default="dummy",
                        type=str)
    parser.add_argument("--enableLogFile", help="Log transitions in file",
                        default=False, action="store_true")
    parser.add_argument("--render", help="Render agent while learning",
                        action="store_true", default=False)
    parser.add_argument("-v", "--verbose", help="Verbose", action="count",
                        default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
