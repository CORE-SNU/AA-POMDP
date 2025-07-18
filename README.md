Anderson-Accelerated Soft POMDP Solvers
====================================================

This repository includes an official Python implementation of the **Anderson-accelerated soft POMDP solvers** presented in **[Anderson acceleration for partially observable Markov decision processes: A maximum entropy approach](https://www.sciencedirect.com/science/article/abs/pii/S0005109824000499)**.



## 1. Requirements
This repository is built upon our previous implementation of [AA-FIB](https://github.com/CORE-SNU/AA-FIB.git), and an official C++ implementation of [SARSOP](https://github.com/AdaCompNUS/sarsop). Our code requires a minimum installation of extra dependencies, such as **numpy** and **scipy**.
We have successfully run our code on Ubuntu 18.04, Python 3.7.4. Note that the codebase is compatible with POMDP problems of `.pomdp` file format from [POMDP.org][POMDP.org], [APPL][APPL].


## 2. Quick Start
First, clone our repository by running:
```
git clone https://github.com/CORE-SNU/AA-POMDP.git
```
### AA-(s)QMDP
We provide a unified python file for testing any algorithms that appear in our paper.
For instance, to use AA-sQMDP solver with the softmax parameter 10, run the following:
```
python main.py --QMDP --safeguard safe_global --max_type logsumexp --softmax_param 10 --do_eval
```


### AA-(s)FIB
To use AA-FIB solver, run the following:
```
cd ./SAA-FIB
python main.py --FIB --safeguard safe_local --safeguard_coeff 100 --do_eval
```

For the reproduction of the results reported in the paper, run the bash script `run_all.sh`.


## 3. Parameter Descriptions
We provide the following arguments for both AA-sQMDP and AA-sFIB experiments:
- env_name : select the POMDP to be solved (among the files under `examples/env`)
- num_trials : number of different initializations
- max_type : standard / mellowmax / logsumexp
- softmax_param: $\lambda$ / $\tau$ for the sFIB
- safeguard : strict (strict safeguard), safe_global (loose safeguard), safe_local (target optimization gain)
- safeguard_coeff : m for target optimization gain
- do_eval : flag to run evaluation (this may induce large computation time)


## 4. Work with other environments
For custom environments, download .pomdp file from [POMDP.org][POMDP.org], [APPL][APPL] to `./examples/env`.

.pomdp file sholud be parsed into .pickle files to be compatible with our code. This can be done by:
```
python convert_pomdp.py POMDP_file
```

To run simulated version of AA-FIB with custon environments, you should get solution of exact version first, and copy them to `./solver_exact` directory.

[arxiv]: https://arxiv.org/abs/2103.15275
[POMDP.org]: http://pomdp.org/examples/
[APPL]: https://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.Repository
