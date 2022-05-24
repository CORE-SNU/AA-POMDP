import pickle, time
from pomdp_accel import POMDP
from environment import Environment
from scipy.sparse import dok_matrix
from multiprocessing import Pool
import pandas as pd
import numpy as np

def load_model(model_name):
    # Load appropriately parsed POMDP environment
    model_file = 'examples/env/' + model_name + '.pomdp'
    pomdp = POMDP(model_file, parse=False)
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
    print('\n[STATUS] successfully loaded pomdp!')

    return pomdp

def sample(pomdp, sample_size):

    print('Type: Simulation-based update')
    environment = Environment(pomdp)
    s_num, a_num, o_num = len(pomdp.states), len(pomdp.actions), len(pomdp.observations)
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

    return O_hat, R_hat

def log(stats, file_name, alg_name, model_name):
    """
    Logging
    """
    with open(file_name, "a") as myfile:
        myfile.write("## Training Result ##\n")
        myfile.write("Model : {}\n".format(model_name))
        myfile.write("Solver : {}\n".format(alg_name))

    # Caculation time logging
    time_mean, time_std = np.mean(stats['total_time']), np.std(stats['total_time'])
    iter_mean, iter_std = np.mean(stats['iters']), np.std(stats['iters'])
    andy_mean, andy_std = np.mean(stats['AA_time']), np.std(stats['AA_time'])
    naa_mean, naa_std = np.mean(stats['naa']), np.std(stats['naa'])
    return_fix_mean, return_fix_std = np.mean(stats['return_fix']), np.std(stats['return_fix'])
    return_rand_mean, return_rand_std = np.mean(stats['return_rand']), np.std(stats['return_rand'])
    
    with open(file_name, "a") as myfile:
        myfile.write("Total iterations  :\t %.2f $\pm$ %.2f \n"%(iter_mean, iter_std))
        myfile.write("# of qAAs used    :\t %.2f $\pm$ %.2f \n"%(naa_mean, naa_std))
        myfile.write("Ellapsed time     :\t %.3f $\pm$ %.3f \n"%(time_mean, time_std))
        myfile.write("AA weight caltime :\t %.3f $\pm$ %.3f \n"%(andy_mean, andy_std))
        myfile.write("Return (fixed)    :\t %.3f $\pm$ %.3f \n"%(return_fix_mean, return_fix_std))
        myfile.write("Return (rand)     :\t %.3f $\pm$ %.3f \n"%(return_rand_mean, return_rand_std))
        myfile.write('\n')  # cf) file.write doesn't automatically type nextline

    """
    if alg_name.split("_")[0] == "AAFIB":
        df = {
            'res_mean': np.mean(np.array(stats['gk']), axis=0),
            'res_std': np.std(np.array(stats['gk']), axis=0),
            'opt_gain_mean': np.mean(np.array(stats['opt_gain']), axis=0),
            'opt_gain_std': np.std(np.array(stats['opt_gain']), axis=0),
            'opt_res_mean': np.mean(np.array(stats['opt_res']), axis=0),
            'opt_res_std': np.std(np.array(stats['opt_res']), axis=0),
        }
    else:
        df = {
            'res_mean': np.mean(np.array(stats['gk']), axis=0),
            'res_std': np.std(np.array(stats['gk']), axis=0)
        }

    df = pd.DataFrame(df)
    df.to_csv(alg_name+"_"+model_name+".csv", index=False)
    """