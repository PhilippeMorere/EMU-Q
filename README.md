# EMU-Q
Exploring by Minimizing Uncertainty of Q values (EMU-Q) as presented in "Bayesian RL for Goal-Only Rewards" at CoRL'18 <link to paper>

## Dependencies
This code is written for python3. The dependencies (pip packages) are:
* numpy
* scipy
* gym
* nlopt
* ghalton
* tqdm

## Running the code
The code entry point is `main.py`. Try run `python3 main.py --help` for available options.
### Running our method
```
python3 main.py --agent=method --sparseGymEnv=MountainCar-v0 --nStep=300 --nEp=10 --nRFF=300 --sigmaS=0.35 --sigmaA=10 -vv
```

### Running RFF-Q
```
python3 main.py --agent=QLearning --gymEnv=MountainCar-v0 --nStep=300 --nEp=30 --nRFF=300 --sigmaS=0.35 --sigmaA=10 -vv
```

## goal-only discrete and continuous gym environments
All goal-only discrete and continuous gym environments presented in the main paper are located in the `gymEnvs` folder. To use them, these environments need to be registered in gym as described in <https://gym.openai.com/docs/#the-registry>.
These environments can then be called from `main.py` with `--gymEnv=SparseMountainCar-v0` for example.
