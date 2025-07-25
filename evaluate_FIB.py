#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""
import pickle
from pomdp_accel import POMDP
from environment import Environment
from offlineSolver import OfflineSolver
from mdpSolver import FIB, A3FIB, PERSEUS
import os
import numpy as np
import sys
import tqdm
from scipy.sparse import csr_matrix, dok_matrix, csc_matrix
import time
from multiprocessing import Pool

if len(sys.argv) == 4:
    """
    Parse arguments
    """
    alg_name = sys.argv[1]   # FIB
    alg_names = [alg_name]
    model_name = sys.argv[2]
    num_runs = int(sys.argv[3]) # 251
    #rand_init = int(sys.argv[4])
    file_name = "result_eval.txt"
    model_file = 'examples/env/' + model_name + '.pomdp'

    """
    POMDP postprocessing
    """
    print("")
    print("#############################")
    print("#    Starting Evaluation    #")
    print("#############################")
    print("")
    # Load POMDP environment
    parse = False
    pomdp = POMDP(model_file, parse=parse)
    if parse:
        pass
    else:
        with open("./examples/env/"+model_name+'_T'+'.pickle', 'rb') as handle:
            pomdp.T = pickle.load(handle)
        with open("./examples/env/"+model_name+'_O'+'.pickle', 'rb') as handle:
            pomdp.O = pickle.load(handle)
        with open("./examples/env/"+model_name+'_R'+'.pickle', 'rb') as handle:
            pomdp.R = pickle.load(handle)
        with open("./examples/env/"+model_name+'_Rconfig'+'.pickle', 'rb') as handle:
            pomdp.flag = pickle.load(handle)

    s_num, a_num, o_num = len(pomdp.states), len(pomdp.actions), len(pomdp.observations)
    print('Sparse T? {}/{}'.format(len(pomdp.T), a_num*s_num*s_num))
    print('Sparse O? {}/{}'.format(len(pomdp.O), a_num*s_num*o_num))
    print('Sparse R? {}/{}'.format(len(pomdp.R), a_num*s_num*s_num*o_num))
    print('Rconfig? : {}'.format(bool(pomdp.flag)))

    pomdp.states = ['%d'%(i) for i in range(s_num)]
    pomdp.actions = ['%d'%(i) for i in range(a_num)]
    pomdp.observations = ['%d'%(i) for i in range(o_num)]
    
    # Build OT Pr(s'|s,o,a) (S X S)
    OT = {o: {a: dok_matrix((s_num, s_num)) for a in range(a_num)} for o in range(o_num)} # O X A entries dict
    for o in range(o_num):
        for key, val in pomdp.T.items():
            a, s, s_ = key
            try:
                OT[o][a][s, s_] = val * pomdp.O[(a, s_, o)]
            except:
                OT[o][a][s, s_] = 0

    for o in range(o_num):
        for a in range(a_num):
            OT[o][a] = OT[o][a].tocsr()
    pomdp.OT = OT
    
    # Build T Pr(s'|s,a) (S X S)
    T = {a: dok_matrix((s_num, s_num)) for a in range(a_num)}
    for key, val in pomdp.T.items():
        a, s, s_ = key
        T[a][s_, s] = val
    for a in range(a_num):
        T[a] = T[a].tocsr()
    pomdp.t = T
    
    # Build O (S X 1)
    O = {a: {o: np.zeros([s_num]) for o in range(o_num)} for a in range(a_num)} # O X A entries dict
    for key, val in pomdp.O.items():
        a, s_, o = key
        O[a][o][s_] = val
    pomdp.o = O

    print('loaded pomdp!')

    """
    Evaluation Loop
    """
    # Create dictionary to append result
    dict_statistics = {i: {r:[] for r in range(2)} for i in alg_names}
    for algn in alg_names:
        print("Evaluating {}...".format(algn))
                
        # Just creating instance to assign value
        if algn == "FIB":
            solver = FIB(pomdp, sample = False)
        elif "A3FIB"  in algn:
            solver = A3FIB(pomdp, sample = False)
        if algn == "PERSEUS":
            solver = PERSEUS(pomdp)
        if algn == "SARSOP":
            solver = PERSEUS(pomdp)  # Both use same class
        
        # Load saved model with designated param (algorithm name)
        with open("solver/"+algn+"_"+model_name+'.pickle', 'rb') as handle:
            b = pickle.load(handle)
        
        eval_start = time.time()
        for r in range(len(b)):
            print('Evaluation process %d/%d'%(r+1, len(b)))
            solver.Q_value = b[r]
            '''
            pool = Pool(processes=11)
            arg = [(False, num_runs), (True, num_runs)]
            result = pool.map(solver.evaluate, arg)
            reward, _ = result[0]
            dict_statistics[algn][0].append(reward)
            reward, _ = result[1]
            dict_statistics[algn][1].append(reward)
            '''
            reward, _ = solver.evaluate(rand_init=False, num_runs=num_runs) # Over 251 runs, due to stochastic system
            dict_statistics[algn][0].append(reward) # Append if ran over multiple repeats
            # reward, _ = solver.evaluate(rand_init=True, num_runs=num_runs) # Over 251 runs, due to stochastic system
            dict_statistics[algn][1].append(reward) # Append if ran over multiple repeats
            #'''
        eval_end = time.time()
        eval_time = eval_end - eval_start
        print('Evaluation time : %.4f'%(eval_time))

    """
    Logging
    """
    with open(file_name, "a") as myfile:
        myfile.write("########    Evaluation Result    ########\n")
        myfile.write("Model : {} \t ".format(model_name))

    for alg_idx, (key_stat, value_stat) in enumerate(dict_statistics.items()):
        reward_mean_fix, reward_std_fix = np.mean(value_stat[0]), np.std(value_stat[0])
        reward_mean_rand, reward_std_rand = np.mean(value_stat[1]), np.std(value_stat[1])

        with open(file_name, "a") as myfile:
            myfile.write("Algorithm : {}\n".format(key_stat))
            myfile.write("Randomized belief return  mean : %.5f  std : %.5f\n"%(reward_mean_rand, reward_std_rand))
            myfile.write("Fixed belief return       mean : %.5f  std : %.5f\n"%(reward_mean_fix, reward_std_fix))
            myfile.write('\n')

    print("")
    print("#############################")
    print("#    Evaluation Finished    #")
    print("#############################")
    print("")
