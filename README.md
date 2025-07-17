Anderson-Accelerated Soft POMDP Solvers
====================================================

This repository includes an official Python implementation of the **Anderson-accelerated soft POMDP solvers** presented in **[Anderson acceleration for partially observable Markov decision processes: A maximum entropy approach](https://www.sciencedirect.com/science/article/abs/pii/S0005109824000499)**.



## 1. Requirements
This repository is built upon our previous implementation of [AA-FIB](https://github.com/CORE-SNU/AA-FIB.git). This requires a minimum installation of extra dependencies, such as **numpy** and **scipy**.
We have successfully run our code on Ubuntu 18.04, Python 3.7.4.


## 2. Quick Start
First, clone our repository by running:
```
git clone https://github.com/CORE-SNU/AA-POMDP.git
```
### AA-sQMDP
To use AA-sQMDP solver, run the following:
```
cd ./SAA-QMDP
python main.py --safeguard safe_local --safeguard_coeff 100 --do_eval
```


### AA-sFIB
To use AA-sFIB solver, run the following:
```
cd ./SAA-FIB
python main.py --safeguard safe_local --safeguard_coeff 100 --do_eval
```



## 3. Solve/Evaluate
We provide the following arguments for AA-sFIB experiments
- env_name : select the POMDP to be solved (among the files under `examples/env`)
- num_trials : number of different initializations
- max_type : standard / mellowmax / logsumexp
- softmax_param: $\lambda$ / $\tau$ for the sFIB
- safeguard : strict (strict safeguard), safe_global (loose safeguard), safe_local (target optimization gain)
- safeguard_coeff : m for target optimization gain
- do_eval : flag to run evaluation (this may induce large computation time)

For the reproduction of the results reported in the paper, run the bash script `run.sh`.
