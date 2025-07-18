import pickle
from pomdp_accel import POMDP
import os
import numpy as np
import sys
import time

if len(sys.argv) == 2:
    model_name = sys.argv[1]  # environment name

    print('')
    print('## Processing Environment... ##')

    # Load POMDP environment
    model_file = 'examples/env/' + model_name + '.pomdp'  # Load pomdp env
    pomdp = POMDP(model_file, parse=True)

    print(model_name)
    print('# of states       : %d' % (len(pomdp.states)))
    print('# of actions      : %d' % (len(pomdp.actions)))
    print('# of observations : %d' % (len(pomdp.observations)))

    s_num, a_num, o_num = len(pomdp.states), len(
        pomdp.actions), len(pomdp.observations)
    flag = pomdp.next_flag

    print('Sparse T? {}/{}'.format(len(pomdp.T), a_num*s_num*s_num))
    print('Sparse O? {}/{}'.format(len(pomdp.O), a_num*s_num*o_num))
    print('Sparse R? {}/{}'.format(len(pomdp.R), a_num*s_num*s_num*o_num))
    print('Reward flag? {}'.format(bool(flag)))

    with open("./examples/env/"+model_name+'_T'+'.pickle', 'wb') as handle:
        pickle.dump(pomdp.T, handle)
    with open("./examples/env/"+model_name+'_O'+'.pickle', 'wb') as handle:
        pickle.dump(pomdp.O, handle)
    with open("./examples/env/"+model_name+'_R'+'.pickle', 'wb') as handle:
        pickle.dump(pomdp.R, handle)
    with open("./examples/env/"+model_name+'_Rconfig'+'.pickle', 'wb') as handle:
        pickle.dump(pomdp.next_flag, handle)

    print('\nSave complete. Loading test\n')
    load_start = time.time()
    with open("./examples/env/"+model_name+'_T'+'.pickle', 'rb') as handle:
        T = pickle.load(handle)
    with open("./examples/env/"+model_name+'_O'+'.pickle', 'rb') as handle:
        O = pickle.load(handle)
    with open("./examples/env/"+model_name+'_R'+'.pickle', 'rb') as handle:
        R = pickle.load(handle)
    with open("./examples/env/"+model_name+'_Rconfig'+'.pickle', 'rb') as handle:
        flag = pickle.load(handle)

    load_end = time.time()
    print('Loading took %.2f secs' % (load_end - load_start))

    s_num, a_num, o_num = len(pomdp.states), len(
        pomdp.actions), len(pomdp.observations)
    print('Sparse T? {}/{}'.format(len(pomdp.T), a_num*s_num*s_num))
    print('Sparse O? {}/{}'.format(len(pomdp.O), a_num*s_num*o_num))
    print('Sparse R? {}/{}'.format(len(pomdp.R), a_num*s_num*s_num*o_num))
    print('Reward flag? {}'.format(bool(flag)))
