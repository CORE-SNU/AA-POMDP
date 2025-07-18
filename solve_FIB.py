#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""
import pickle
from pomdp_accel import POMDP
from environment import Environment
from offlineSolver import OfflineSolver
from mdpSolver import  FIB, A3FIB
from softmdpSolver import  softFIB, softA3FIB
import os
import numpy as np
from numpy import linalg as LA
import sys
import matplotlib.pyplot as plt
import time
from scipy.sparse import csr_matrix, dok_matrix, csc_matrix
#from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

if len(sys.argv) == 5 or len(sys.argv) == 6:
    """
    Parse arguments
    """
    alg_name = sys.argv[1]  # FIB
    alg_names = [alg_name]
    model_name = sys.argv[2]    # environment name
    repeat_number = int(sys.argv[3]) # number of different initializations
    precision = float(sys.argv[4]) # update threshold to stop FIB update
    if len(sys.argv) == 6:
        sample_size = int(sys.argv[5]) # Sampling number
        sample = True
    else:
        sample = False
    file_name = "result.txt"
    
    """
    POMDP postprocessing
    """
    print('')
    print('###########################')
    print('#     Starting Setups     #')
    print('###########################')
    print('')
    print("alg : {} \t model : {}".format(alg_name, model_name))
    
    # Load POMDP environment
    model_file = 'examples/env/' + model_name + '.pomdp'    # Load pomdp env
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
    '''
    print('Sparse T? {}/{}'.format(len(pomdp.T), a_num*s_num*s_num))
    print('Sparse O? {}/{}'.format(len(pomdp.O), a_num*s_num*o_num))
    print('Sparse R? {}/{}'.format(len(pomdp.R), a_num*s_num*s_num*o_num))
    print('Rconfig? : {}'.format(bool(pomdp.flag)))
    '''

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
    print('loaded pomdp!')

    """
    Sampling (s,a,r,s_,o)
    """
    if sample:
        print('Type: Simulation-based update')
        environment = Environment(pomdp)
        sample_start = time.time()

        pool = Pool(processes=11)
        arg = [(s, a, sample_size) for s in range(s_num) for a in range(a_num)]
        result = pool.map(environment.sample, arg)

        O_hat = {o: {a: dok_matrix((s_num, s_num)) for a in range(a_num)} for o in range(o_num)} # O X A entries dict
        R_hat = dok_matrix((a_num, s_num))
        
        sample_time = time.time() - sample_start
        print('Sampling finished! elapsed time: %.2f'%(sample_time))

        for key, val in result:
            s, a = key
            for (s_, o, r) in val:
                O_hat[o][a][s, s_] += 1/sample_size
                R_hat[a, s] += r/sample_size
        
        for o in range(o_num):
            for a in range(a_num):
                O_hat[o][a] = O_hat[o][a].tocsr()
        R_hat = R_hat.tocsr()
    else:
        print('Type: Exact update')

    with open("solver_exact/FIB_"+model_name+'.pickle', 'rb') as handle:
        FIB_sol = pickle.load(handle)
        FIB_sol = FIB_sol[1]

    """
    FIB Solver Loop
    """
    print('')
    print('###########################')
    print('#      Solving POMDP      #')
    print('###########################') 
    # Create dictionary to store result
    dict_statistics = {i: {s:[] for s in range(7)} for i in alg_names}
    flag = [True]
    for r in range(repeat_number):
        #pomdp = POMDP(model_file)  # Use POMDP wrapper to read pomdp file
        print('')
        print('Solving %d/%d POMDPs'%(r+1, repeat_number))
        for init in flag:
            initial_vector = None
            
            # Solve with baseline FIB
            if sys.argv[1] == "FIB":
                # Calling solver will automatically solve pomdps
                solver = softFIB(pomdp, precision, sample = sample)
                if sample:
                    solver.O_hat, solver.R_hat = O_hat, R_hat
                solver.update()
            
            # Log solver output for FIB
            alg_orig = alg_names[0]
            dict_statistics[alg_orig][0].append(solver.total_time)  # Total time for FPI
            dict_statistics[alg_orig][1].append(solver.time_step)   # Num of iterations
            dict_statistics[alg_orig][2].append(0) 
            dict_statistics[alg_orig][3].append(solver.Q_value)
            dict_statistics[alg_orig][4].append(0)
            if len(solver.res) > 2000: solver.res = solver.res[:2000]
            res = solver.res[-1] * np.ones(2000)
            for i, r in enumerate(solver.res): res[i] = r
            dict_statistics[alg_orig][5].append(res)
            g = np.reshape(solver.Q_value, -1)-np.reshape(FIB_sol, -1)
            dict_statistics[alg_orig][6].append(LA.norm(g))
            initial_vector = solver.initial_vector # To use same inital vector in A3FIB
    
    """
    Save Q-value
    Since appending, this should have repeat_number*A numbers of vectors
    """
    for algn in alg_names:
        with open("solver/"+algn+"_"+model_name+'.pickle', 'wb') as handle:
            # Save result from dict_statistics
            pickle.dump(dict_statistics[algn][3], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        df = {'mean': np.mean(np.array(dict_statistics[algn][5]), axis=0),
            'std': np.std(np.array(dict_statistics[algn][5]), axis=0)}
        df = pd.DataFrame(df)
        df.to_csv(algn+"_"+model_name+'.csv', index=False)

    """
    Logging
    """
    with open(file_name, "a") as myfile:
        myfile.write("## Training Result ##\n")
        myfile.write("Model : {} \t ".format(model_name))

    for alg_idx, (key_stat, value_stat) in enumerate(dict_statistics.items()):
        # Caculation time logging
        algo_name = sys.argv[1] if not alg_idx else algo
        time_mean, time_std = np.mean(value_stat[0]), np.std(value_stat[0])
        iter_mean, iter_std = np.mean(value_stat[1]), np.std(value_stat[1])
        sample_mean, sample_std = np.mean(value_stat[2]), np.std(value_stat[2])
        naa_mean, naa_std = np.mean(value_stat[4]), np.std(value_stat[4])
        FIB_err_mean, FIB_err_std = np.mean(value_stat[6]), np.std(value_stat[6])
        with open(file_name, "a") as myfile:
            myfile.write("Algorithm : {}\t\n".format(sys.argv[1]))
            myfile.write("Ellapsed time     \t mean : %.5f \t std : %.5f \n"%(time_mean, time_std))
            myfile.write("Total iterations  \t mean : %.5f \t std : %.5f \n"%(iter_mean, iter_std))
            if sample:
                myfile.write("Sampling param    \t time : %.5f \t num : %.5f \n"%(sample_time, sample_size))
            myfile.write("F_hat error       \t mean : %.5f \t std : %.5f \n"%(FIB_err_mean, FIB_err_std))
            #myfile.write("# of qAAs used    \t mean : %.5f \t std : %.5f \n"%(naa_mean, naa_std))
            myfile.write('\n')  # cf) file.write doesn't automatically type nextline


    print('')
    print('###########################')
    print('# Finished Solving POMDPs #')
    print('###########################')
    print('')

