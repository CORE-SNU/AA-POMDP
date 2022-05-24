Safeguarded Anderson Acceleration for Regularized Bellman Iterations
====================================================

This repository includes an official python implementation of the **Safeguarded Anderson Acceleration** presented in **Safeguarded Anderson Acceleration for Regularized Bellman Iterations**



## 1. Requirements
Our implementation is based on [AA-FIB](https://github.com/CORE-SNU/AA-FIB.git) and [RAA](https://github.com/shiwj16/raa-drl), and followings must be installed to run our implementation:
- **Python (>=3.7)**
- **[PyTorch][Pytorch]**
- **[OpenAI Gym <= 0.12.1][Gym]**
- **[Atari-py](https://github.com/openai/atari-py.git)**
- **ale-py**
- **[Mujoco](https://github.com/openai/mujoco-py#install-mujoco)**
- **[Mujoco-py](https://github.com/openai/mujoco-py.git)**

For detailed installation instructions for Atari and Mujoco, follow prerequisites from [Dopamine](https://github.com/google/dopamine.git).




## 2. Quick Start
First, clone our repository with:
```
git clone https://github.com/CORE-SNU/MPC-PEARL.git
```
### 2.1 sAA-FIB
Try solving simple example problem under the `./SAA-FIB` directory with:
```
python main.py --safeguard safe_local --safeguard_coeff 100 --do_eval
```
### 2.1 sAA-DRL
Train sAA-SAC with the following command under the `./SAA-DRL/SAA-SAC` directory:
```
python main.py --env_name Hopper-v3 --reg_scale 0.1 --theta_thres 0.96 --seed 101
```



## 3. Solve/Evaluate
### 3.1 Run sAA-FIB Experiment
We provide below arguments for sAA-FIB experiments
- env_name : select the POMDP to be solved (among the files under examples/env)
- num_trials : number of different initializations
- max_type : standard / mellowmax / logsumexp
- softmax_param: lambda / tau for the regularized FIB
- safeguard : strict (strict safeguard), safe_global (loose safeguard), safe_local (target optimization gain)
- safeguard_coeff : m for target optimization gain
- do_eval : flag to run evaluation (this may induce large computation time)


### 3.2 Run sAA-DRL Experiments
We provide below arguments for sAA-FIB experiments
- env_name : select the POMDP to be solved (among the files under examples/env)
- agent_name : regularization type (RAA and A3 corresponds to cAA and uAA, respectively).
- reg_scale : regularization coefficient for AA

The training history is stored under the `./logs` directory.
Moreover, we provide random seeds for reproduction.
- For discrete control tasks, we used seeds 101 ~ 105
- For continuous control tasks, we used seeds 101 ~ 108