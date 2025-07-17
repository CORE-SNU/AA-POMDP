#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""
import click
from utils import *
from AAQMDP import AAQMDP, SARSOP
import numpy as np

def solve_AA_QMDP(solver, num_trials, do_eval):
    """
    A3QMDP Solver Loop
    """
    stats = {
        'total_time':[],
        'AA_time':[],
        'iters':[],
        'naa':[],
        'QMDP_vec':[],
        'gk':[],
        'opt_gain':[],
        'opt_res':[],
        'return_fix':[],
        'return_rand':[]
    }

    for r in range(num_trials):
        # Solve
        print('Solving %d/%d POMDPs'%(r+1, num_trials))
        solver.solve()
        
        stats['total_time'].append(solver.total_time)
        stats['iters'].append(solver.time_step)
        stats['QMDP_vec'].append(solver.total_time)

        res = solver.res[-1] * np.ones(solver.max_iter)
        for i, r in enumerate(solver.res): res[i] = r
        stats['gk'].append(res)

        if not solver.FIB:
            stats['AA_time'].append(solver.andy)
            stats['naa'].append(solver.naa)
            opt_gain = solver.opt_gain[-1] * np.ones(solver.max_iter)
            opt_res = solver.opt_res[-1] * np.ones(solver.max_iter)
            for i, r in enumerate(solver.opt_gain): opt_gain[i] = r
            for i, r in enumerate(solver.opt_res): opt_res[i] = r
            stats['opt_gain'].append(opt_gain)
            stats['opt_res'].append(opt_res)   

        # Evaluation 
        if do_eval:
            reward, _ = solver.evaluate(rand_init=True, num_runs=100)
            stats['return_rand'].append(reward)
            reward, _ = solver.evaluate(rand_init=False, num_runs=100)
            stats['return_fix'].append(reward)

    return stats


@click.command()
@click.option('--SARSOP', is_flag=True, default=False, help='evaluate SARSOP if turned on')
@click.option('--QMDP', is_flag=True, default=False, help='solve FIB if turned on')
@click.option('--sample', is_flag=True, default=False, help='solve sampled version of FIB if turned on')
@click.option('--sample_size', default=10, help='number of samples for the sampled version')
@click.option('--env_name', default="under_water")
@click.option('--max_iter', default=4000)
@click.option('--num_trials', default=100)
@click.option('--do_eval', is_flag=True, default=False, help='evaluate stored value if turned on')
#@click.option('--eval_per_trial', default=100)
#@click.option('--eval_length', default=100)
@click.option('--eps', default=1e-6, help='Termination threshold')
@click.option('--max_type', default="standard", help='FIB parameter') # standard / mellowmax / logsumexp
@click.option('--softmax_param', default=0.1, help='softmax parameter') # Find that preserves performance
@click.option('--max_mem', default=16, help='AA parameter') # 4, 8, 12, 16
@click.option('--eta', default=1e-16, help='AA regularization parameter') # 1e-8, 1e-16, 1e-32
@click.option('--safeguard', default="safe_local", help='AA parameter') # safe_local, safe_global, strict, opt_gain
@click.option('--safeguard_coeff', default=0.8, help='AA parameter') # m or opt_bar when choosing 
@click.option('--filename', default="./Underwater_final.txt", help='location to store statistics')
def main(
    sarsop,
    fib,
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
    filename ):

    print('')
    print('###########################')
    print('#     Starting Setups     #')
    print('###########################')
    pomdp = load_model(env_name)
    if sarsop:
        alg_name = "SARSOP"
        solver = SARSOP(pomdp)
        
        print('')
        print("alg : {} \t model : {}".format(alg_name, env_name))
        
        # Load saved model with designated param (algorithm name)
        with open("solver/"+alg_name+"_"+env_name+'.pickle', 'rb') as handle:
            b = pickle.load(handle)
        
        stats = {'return_fix':[], 'return_rand':[]}
        for r in range(len(b)):
            print('Evaluation process %d/%d'%(r+1, len(b)))
            solver.Q_value = b[r]
            reward, _ = solver.evaluate(rand_init=False, num_runs=100)
            stats['return_fix'].append(reward)
            reward, _ = solver.evaluate(rand_init=True, num_runs=100)
            stats['return_rand'].append(reward)
        return_fix_mean, return_fix_std = np.mean(stats['return_fix']), np.std(stats['return_fix'])
        return_rand_mean, return_rand_std = np.mean(stats['return_rand']), np.std(stats['return_rand'])
    
        with open(filename, "a") as myfile:
            myfile.write("## Training Result ##\n")
            myfile.write("Model : {}\n".format(env_name))
            myfile.write("Solver : {}\n".format(alg_name))
            myfile.write("Return (fixed)    :\t %.3f $\pm$ %.3f \n"%(return_fix_mean, return_fix_std))
            myfile.write("Return (rand)     :\t %.3f $\pm$ %.3f \n"%(return_rand_mean, return_rand_std))
            myfile.write('\n')  # cf) file.write doesn't automatically type nextline
    
    else:
        print('')
        alg_name = "QMDP"
        if not fib:
            alg_name = "AA" + alg_name + "_" + str(max_mem) + "_" + max_type + "_" + str(softmax_param) + "_"+ safeguard + "_" + str(safeguard_coeff)
        print("alg : {} \t model : {}".format(alg_name, env_name))
        solver = AAFIB(
            pomdp,
            fib,
            sample,
            max_iter, 
            eps,
            softmax_param,
            max_mem,
            eta,
            safeguard_coeff,
            max_type = max_type,
            safeguard = safeguard
        )
        if sample:
            solver.O_hat, solver.R_hat = sample(sample_size)
        
        print('')
        print('###########################')
        print('#      Solving POMDP      #')
        print('###########################')
        print('')
        stats = solve_AA_QMDP(solver, num_trials, do_eval)
        log(stats, filename, alg_name, env_name)
        
        print('')
        print('###########################')
        print('# Finished Solving POMDPs #')
        print('###########################')
        print('')


if __name__ == "__main__":
    main()
