import sys
from tqdm import tqdm
from rlutils.EnvGlue import AbsEnvGlue
from rlutils.Agents import AbsAgent
from rlutils.Policies import AbsPolicy


class EnvRunner:
    def __init__(self, envGlue, policy, agent, tm=None, verbose=1,
                 updateAgent=True):
        self.tm = tm  # transition memory
        self.verbose = verbose

        # Environment
        if not isinstance(envGlue, AbsEnvGlue):
            print(envGlue, "/", AbsEnvGlue)
            raise ValueError("Environment glue must be of type AbsEnvGlue")
        self.envGlue = envGlue

        # Policy
        if not isinstance(policy, AbsPolicy):
            raise ValueError("Policy must be of type AbsPolicy")
        self.policy = policy

        # Agent
        if not isinstance(agent, AbsAgent):
            raise ValueError("Agent must be of type AbsAgent")
        self.agent = agent
        self.updateAgent = updateAgent

    def run(self, nEp, nStep, stopAtPosReturn=False):
        if self.verbose == 1:
            gen = tqdm(range(nEp))
        else:
            gen = range(nEp)
        for i in gen:
            if self.verbose > 1:
                sys.stdout.write("Episode {}".format(i+1))
                sys.stdout.flush()

            # Run episode
            epRet, lastStepDone, nStepFinish, lastR = self.runEpisode(nStep)

            # Save transitions
            if self.tm:
                self.tm.endEpisode()

            if stopAtPosReturn:
                if lastStepDone and nStepFinish < nStep and lastR >= 0:
                    break

            # Potential update at the end of the episode
            if self.updateAgent:
                self.agent.endOfEpUpdate()

        if self.verbose > 1:
            print("Finished")

    def runEpisode(self, nStep):
        s = self.envGlue.resetEnv()
        ret = 0
        done = False
        for i in range(nStep):
            a = self.policy.pick(s)
            sp, r, done = self.envGlue.stepEnv(a)
            ret += r
            if self.verbose > 2:
                print("\nTransition:\ts:{}\n\t\ta:{}\n\t\tr:{}\n\t\ts':{}".
                      format(s, a, r, sp))

            if self.tm:
                self.tm.addStep(s, a, r, sp)

            if self.updateAgent:
                self.agent.update(s, a, r, sp)

            s = sp
            if done:
                break
        if self.verbose > 1:
            print("... finished in", i+1, "steps with return", ret)
        return ret, done, i+1, r
