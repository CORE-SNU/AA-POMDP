
from environment import Environment
from offlineSolver import OfflineSolver
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import time
import click
from utils import *
import math
from scipy.sparse import csr_matrix, dok_matrix, csc_matrix, identity, vstack, kron
from scipy.sparse import linalg as la
from scipy.special import logsumexp
import pickle


class MeanQMDP(OfflineSolver):
    def __init__(self, 
        pomdp,
        sample,
        eps
        ):

        """
        Initialization
        """
        super(MeanQMDP, self).__init__(pomdp, eps)
        self.s_num = len(self.pomdp.states)
        self.a_num = len(self.pomdp.actions)
        self.o_num = len(self.pomdp.observations)
        self.sample = sample
        self.eps = eps
        
        self.random_policy = np.random.randint(self.a_num, size=self.s_num)

    def solve(self):
        # Initialize data
        R = dok_matrix((self.a_num, self.s_num))
        if self.pomdp.flag:
            for key, val in self.pomdp.T.items():
                a, s, s_ = key
                try:
                    R[a, s] += val * self.pomdp.R[(a ,s_)]
                except:
                    pass
        else: 
            for key, val in self.pomdp.R.items():
                R[key[0], key[1]] = val
        self.rewards = R.toarray()

        # Estimate transitions if sample
        if self.sample:
            self.T_hat, self.O_hat, self.R_hat = simulate(self.pomdp, 10)

        reward_min, reward_max = np.min(self.rewards) , np.max(self.rewards)
        if reward_min == reward_max:
            if reward_max < 0:
                reward_max = 0
            else:
                reward_min = 0
        
        

        tran_mat = vstack([self.pomdp.tran[a] for a in range(self.a_num)])
        tran_mat_rnd = kron(np.ones(self.a_num) / self.a_num, tran_mat)
        self.reward_vec = np.reshape(self.rewards, newshape=(-1,))
        self.A = identity(self.a_num*self.s_num) - self.pomdp.discount * tran_mat_rnd
     

        # matrix of shape (|S||A|, |S||A|)
        
        # self.total_time = 0.
        start = time.time()
        self.Q_value = np.reshape(la.spsolve(self.A, self.reward_vec), newshape=(self.a_num, self.s_num))
        self.total_time = time.time() - start

    def update_RAA(self):
        print("Need implementation?")


    def chooseAction(self, cur_belief):
        return np.argmax(np.matmul(self.Q_value, cur_belief.T))


    def getValue(self, belief):
        value = np.max(np.matmul(self.Q_value, belief.T))
        return math.floor(value * 100) / 100.0


def solve_MeanQMDP(solver, num_trials, do_eval):
    """
    A3FIB Solver Loop
    """
    stats = {
        'total_time':[],
        'return_fix':[],
        'return_rand':[]
    }

    for r in range(num_trials):
        # Solve
        print('Solving %d/%d POMDPs'%(r+1, num_trials))
        solver.solve()
        
        stats['total_time'].append(solver.total_time)
 
        # Evaluation 
        if do_eval:
            reward, _ = solver.evaluate(rand_init=True, num_runs=100)
            stats['return_rand'].append(reward)
            reward, _ = solver.evaluate(rand_init=False, num_runs=100)
            stats['return_fix'].append(reward)

    return stats



def log_simple(stats, file_name, alg_name, model_name):
    """
    Logging
    """
    with open(file_name, "a") as myfile:
        myfile.write("## Training Result ##\n")
        myfile.write("Model : {}\n".format(model_name))
        myfile.write("Solver : {}\n".format(alg_name))

    # Caculation time logging
    time_mean, time_std = np.mean(stats['total_time']), np.std(stats['total_time'])

    return_fix_mean, return_fix_std = np.mean(stats['return_fix']), np.std(stats['return_fix'])
    return_rand_mean, return_rand_std = np.mean(stats['return_rand']), np.std(stats['return_rand'])
    
    with open(file_name, "a") as myfile:
        myfile.write("Elapsed time     :\t %.3f $\pm$ %.3f \n"%(time_mean, time_std))
        myfile.write("Return (fixed)    :\t %.3f $\pm$ %.3f \n"%(return_fix_mean, return_fix_std))
        myfile.write("Return (rand)     :\t %.3f $\pm$ %.3f \n"%(return_rand_mean, return_rand_std))
        myfile.write('\n')  # cf) file.write doesn't automatically type nextline



@click.command()
@click.option('--SARSOP', is_flag=True, default=False, help='evaluate SARSOP if turned on')
@click.option('--FPI', is_flag=True, default=False, help='solve FPI if turned on')
@click.option('--QMDP', is_flag=True, default=False, help='default state-space method is set to FIB, turn on')
@click.option('--sample', is_flag=True, default=False, help='solve sampled version of the state-space method if turned on')
@click.option('--sample_size', default=10, help='number of samples for the sampled version')
@click.option('--env_name', default="sunysb")
@click.option('--max_iter', default=4000)
@click.option('--num_trials', default=100)
@click.option('--timeout', default='1')
@click.option('--do_eval', is_flag=True, default=False, help='evaluate stored value if turned on')
#@click.option('--eval_per_trial', default=100)
#@click.option('--eval_length', default=100)
@click.option('--eps', default=1e-6, help='Termination threshold')
@click.option('--max_type', default="standard", help='FPI parameter') # standard / mellowmax / logsumexp
@click.option('--softmax_param', default=0.1, help='softmax parameter') # Find that preserves performance
@click.option('--max_mem', default=16, help='AA parameter') # 4, 8, 12, 16
@click.option('--eta', default=1e-16, help='AA regularization parameter') # 1e-8, 1e-16, 1e-32
@click.option('--safeguard', default="safe_local", help='AA parameter') # safe_local, safe_global, strict, opt_gain
@click.option('--safeguard_coeff', default=0.8, help='AA parameter') # m or opt_bar when choosing 
@click.option('--filename', default="./sunysb_fib.txt", help='location to store statistics')
def main(
    sarsop,
    fpi,
    qmdp,
    sample,
    sample_size,
    env_name, 
    max_iter, 
    num_trials,
    do_eval,
    #eval_per_trial,
    #eval_length,
    eps,
    max_type,
    softmax_param,
    max_mem,
    eta,
    safeguard,
    safeguard_coeff,
    timeout,
    filename ):

    print('')
    print('###########################')
    print('#     Starting Setups     #')
    print('###########################')
    pomdp = load_model(env_name)

    print('')
    alg_name = "MeanQMDP"

    solver = MeanQMDP(
        pomdp,
        sample,
        eps,
    )
    if sample:
        alg_name = alg_name + "_sim"
    print("alg : {} \t model : {}".format(alg_name, env_name))

    print('')
    print('###########################')
    print('#      Solving POMDP      #')
    print('###########################')
    print('')
    stats = solve_MeanQMDP(solver, num_trials, do_eval)
    print(stats)
    log_simple(stats, filename, alg_name, env_name)
    
    print('')
    print('###########################')
    print('# Finished Solving POMDPs #')
    print('###########################')
    print('')
    



if __name__ == "__main__":
    main()
