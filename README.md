Anderson-Accelerated Soft POMDP Solvers
====================================================

This repository includes an official Python implementation of the **Anderson-accelerated soft POMDP solvers** presented in **[Anderson acceleration for partially observable Markov decision processes: A maximum entropy approach](https://www.sciencedirect.com/science/article/abs/pii/S0005109824000499)**.



## 1. Requirements
Our implementation is built upon [AA-FIB](https://github.com/CORE-SNU/AA-FIB.git). Our implementation requires a minimum installation of extra dependencies, such as **numpy** and **scipy**.


For detailed installation instructions for Atari and Mujoco, follow the prerequisites from [Dopamine](https://github.com/google/dopamine.git).




## 2. Quick Start
First, clone our repository by running:
```
git clone https://github.com/CORE-SNU/MPC-PEARL.git
```
### AA-sFIB
Try solving a simple example problem under the `./SAA-FIB` directory with:
```
python main.py --safeguard safe_local --safeguard_coeff 100 --do_eval
```



## 3. Solve/Evaluate
We provide the following arguments for sAA-FIB experiments
- env_name : select the POMDP to be solved (among the files under examples/env)
- num_trials : number of different initializations
- max_type : standard / mellowmax / logsumexp
- softmax_param: $\lambda$ / $\tau$ for the regularized FIB
- safeguard : strict (strict safeguard), safe_global (loose safeguard), safe_local (target optimization gain)
- safeguard_coeff : m for target optimization gain
- do_eval : flag to run evaluation (this may induce large computation time)



The training history is stored under the `./logs` directory.
Moreover, we provide random seeds for reproduction.
- For discrete control tasks, we used seeds 101 ~ 105
- For continuous control tasks, we used seeds 101 ~ 108
