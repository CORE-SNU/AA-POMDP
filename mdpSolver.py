'''
FIB & A3FIB Class
'''
from environment import Environment
from offlineSolver import OfflineSolver
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import time
import math
from multiprocessing import Pool
import tqdm
from scipy.sparse import csr_matrix, dok_matrix, csc_matrix, identity
from scipy.sparse import linalg as la
from tqdm import tqdm
import pickle

MIN = -np.inf
file_name = "returns.txt"

class FIB(OfflineSolver):
    def __init__(self, pomdp, precision = .001, sample = False, initial_vector = None):
        
        """
        Initialization
        """        
        super(FIB, self).__init__(pomdp, precision)
        self.normlist, self.res = [], []
        self.sample = sample

        """
        Convert T, O, R to sparse matrix
        """
        self.s_num, self.a_num, self.o_num = len(self.pomdp.states), len(self.pomdp.actions), len(self.pomdp.observations)
        
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

        reward_min, reward_max = np.min(self.rewards) , np.max(self.rewards)
        if reward_min == reward_max:
            if reward_max < 0:
                reward_max = 0
            else:
                reward_min = 0
        
        reward_min, reward_max = reward_min / (1-self.pomdp.discount), reward_max / (1-self.pomdp.discount) # Value function max/min
        
        # Initialize alpha vector (A X S) - horizontal stack
        # Okay with np (technically okay...)
        self.initial_vector = np.random.uniform(reward_min, reward_max,[len(self.pomdp.actions), len(self.pomdp.states)]) #if init else initial_vector
        self.Q_value = self.initial_vector
        self.environment = Environment(self.pomdp)
        if self.sample and self.s_num < 100:
            self.sample_size = 5
        elif self.sample and self.s_num < 1000:
            self.sample_size = 20
        elif self.sample and self.s_num < 10000:
            self.sample_size = 100
        elif self.sample and self.s_num >= 10000:
            print('Too large state space! not for sample update!')
            quit()
        
        '''
        # Reward given as R(s,a,s',o)(?), so reconstruct reward table with maximum expected reward
        for s_index, state in enumerate(self.pomdp.states):
            for a_index, action in enumerate(self.pomdp.actions):
                self.rewards[a_index, s_index] = np.max(np.dot(self.pomdp.T[a_index, s_index, :], self.pomdp.R[a_index, s_index, :, :]))
        '''

    def update(self):
        """
        Main Update for FIB
        """
        time_step = 0
        max_abs_reward = np.max(np.abs(self.rewards))
        
        if True:
            print('FIB update on progress...')
            
            alpha_start = time.time()
        
            # Iterative FIB update
            while True:
                compare_v = self.Q_value.copy()
                if self.sample:
                    self.updateQValueFromSampleValues()
                else:
                    self.updateQValueFromValues()
                # Log alpha change in norm
                vv1, vv2 =  self.Q_value, compare_v
                norm_diff = LA.norm((vv1-vv2).reshape(-1))
                self.normlist.append(norm_diff)
                time_step += 1
                
                '''
                if time_step % 100 == 0:
                    #reward = self.evaluate(251)
                    #self.returns.append(reward)
                    print('{} iteration,  residual : {:.4f}'.format(time_step, norm_diff))
                    #print(self.Q_value)
                '''
                # Check stopping condition - maximum difference in vector
                if self.get_vbq(self.Q_value.copy(), compare_v.copy(), self.precision):
                    break
            
            '''
            # Return logging for experiment 
            with open(file_name, "a") as myfile:
                myfile.write("## FIB reward ##")
                myfile.write("timestep\tavg.return\n")
                for idx, i in enumerate(self.returns):
                    myfile.write("%d\t%.4f\n"%(idx, i))
            '''

            alpha_end = time.time()
            self.time_step = time_step  # Number of steps for FIB to converge
            self.total_time = alpha_end - alpha_start
    
    def chooseAction(self, cur_belief):
        return np.argmax(np.matmul(self.rewards, cur_belief.T) + self.pomdp.discount * np.matmul(self.Q_value, cur_belief.T))

    def getValue(self, belief):
        value = np.max(np.matmul(self.Q_value, belief.T))
        return math.floor(value * 100) / 100.0

    def updateQValueFromValues(self):
        Q_value, Q = self.Q_value.copy(), self.Q_value.copy()
        # Tabular representation
        for a_index, action in enumerate(self.pomdp.actions):
            v = np.zeros([len(self.pomdp.states)])
            for o_index, obs in enumerate(self.pomdp.observations):
                # Need to convert this!
                #q = np.multiply(self.pomdp.O[(a_index, :, o_index)],self.pomdp.T[(a_index, :, :)])
                v += np.max(np.array([self.pomdp.OT[o_index][a_index].dot(Q[i,:]) for i in range(Q.shape[0])]), axis = 0)          
            Q_value[a_index, :] = self.rewards[a_index, :] + self.pomdp.discount * v

        self.Q_value = Q_value
        
        # Append residual
        g = np.reshape(Q,-1)-np.reshape(Q_value,-1)
        #print(g)
        self.res.append(LA.norm(g))

    def updateQValueFromSampleValues(self):
        Q_value, Q = self.Q_value.copy(), self.Q_value.copy()

        for a in range(self.a_num):
            v = np.zeros([self.s_num])
            for o in range(self.o_num):
                #v += np.max(np.array([O_hat.dot(Q[i,:]) for i in range(self.a_num)]))
                v += np.max(np.array([self.O_hat[o][a].dot(Q[i,:]) for i in range(self.a_num)]), axis = 0)          
            Q_value[a, :] = self.R_hat[a, :] + self.pomdp.discount * v

        self.Q_value = Q_value
        #Q_value = self.Q_value.copy()
        
        # Append residual
        g = np.reshape(Q,-1)-np.reshape(Q_value,-1)
        #print(g)
        self.res.append(LA.norm(g))

    def get_vbq(self,qv1,qv2,eps):
        delta = MIN
        v1, v2 = np.max(qv1,axis = 0), np.max(qv2,axis = 0)
        x = v1-v2
        delta = max(abs(x.min()), abs(x.max()))
        return delta < eps

class A3FIB(OfflineSolver):
    def __init__(self, pomdp, precision = 1e-1, mem = 5, lamb_ = 1e-6, max_iter = 1350, ada_reg = True, sample = False, initial_vector = None):
        """
        Initialization
        """
        super(A3FIB, self).__init__(pomdp, precision)
        # A3 parameters
        self.normlist = []  # Difference between updated vector
        self.res = []   # Residuals
        self.alpha = [] # Coefficient vector
        self.returns = []
        self.mem, self.lamb, self.max_iter = mem, lamb_, max_iter
        self.s_num, self.a_num, self.o_num = len(self.pomdp.states), len(self.pomdp.actions), len(self.pomdp.observations)
        self.sample = sample
               
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

        reward_min, reward_max = np.min(self.rewards) , np.max(self.rewards)
        if reward_min == reward_max:
            if reward_max < 0:
                reward_max = 0
            else:
                reward_min = 0
        reward_min, reward_max = reward_min / (1-self.pomdp.discount), reward_max / (1-self.pomdp.discount) # Value function max/min
        
        self.initial_vector = np.random.uniform(reward_min,reward_max,[len(self.pomdp.actions), len(self.pomdp.states)]) #if init else initial_vector
        self.Q_value = self.initial_vector
        self.ada_reg = True
        
    def update(self):
        """
        Main Update
        """
        #safeguard
        eps, D = 1e-6, 1e6
        naa, Raa = 0, 0
        g0, safeguard = None, True
        R = self.max_iter / 10

        time_step = 1
        max_abs_reward = np.max(np.abs(self.rewards))

        Y, Y_list, S, S_list, Qc, Q_old, delta_trans = [], [], [], [], [], [], []
        self.Q_value, g0 = self.updateQValueFromValues()
        Qlast= self.Q_value.reshape(-1,1)
        Q_old = self.Q_value.reshape(-1)
        Qc.append(self.Q_value)
        delta_trans.append(g0)
        gk_old = g0
        self.OT = self.pomdp.OT

        if True:
            alpha_start = time.time()
            andy = 0
            # Need to store residual norms?
            while True:
                
                if self.sample:
                    Qc_add, gk = self.updateQValueFromSampleValues()
                else:
                    Qc_add, gk = self.updateQValueFromValues()  # Qc_add - FIB updated / gk - residual(1D vector)
                
                #print('Shape of gk : {}'.format(gk))
                Qc.append(Qc_add)
                Y_list.append(gk - gk_old)
                gk_old = gk
                self.res.append(LA.norm(gk))
                compare_v = self.Q_value.copy()
                
                # Stack memory & dropout old memory
                if len(Qc) > self.mem:
                    Qc.pop(0)
                    Y_list.pop(0)

                # AA weight calculation
                and_start = time.time()
                
                Y = np.transpose(np.array(Y_list))
                if time_step:
                    S = np.diff(Qlast)
                else:
                    S = np.transpose(np.array(S_list))
                
                # Adaptive regularization
                # self.lamb : eta in paper
                lamb = self.lamb * (LA.norm(S, 'fro')**2 + LA.norm(Y, 'fro')**2)
                delta = Y
                delta_t = np.transpose(delta)
                reg_identity = lamb*np.identity(len(delta_t))
                delta_inverse = inv(np.add(np.matmul(delta_t,delta), reg_identity))
                gamma_ = np.matmul(np.matmul(delta_inverse, delta_t),gk)
                
                #print(np.matmul(delta_inverse, delta_t).shape)
                gamma_diff = np.diff(gamma_, n=1)
                
                # Recover weight from gammas
                alpha_solution = np.concatenate(([gamma_[0]], gamma_diff, [1-gamma_[-1]]))
                self.alpha.append(alpha_solution)
                andy += time.time() - and_start  # Elapsed time for AA weight caculation
                
                # qAA - [A X S], horizontal stack of value function vector
                Qanderson = np.zeros([len(self.pomdp.actions), len(self.pomdp.states)])
                for cidx, coeff in enumerate(alpha_solution):
                    Qanderson += coeff*Qc[cidx]

                # Safeguard
                # naa : number of aa estimates used in total iteration
                if safeguard or Raa >= R:
                    if LA.norm(gk) <= D*LA.norm(g0)*(naa/R + 1)**(-(1+eps)):
                        naa += 1
                        Raa = 1
                        safeguard = False
                    else:
                        Qanderson = Qc[-1]  # Do not use AA update version
                        Raa = 0
                else:
                    naa +=1
                    Raa +=1

                self.Q_value = Qanderson

                #Qlast = np.append(Qlast, Qanderson.reshape(-1,1), axis=1)
                #print(Qlast.shape)
                
                Q_new = Qanderson.reshape(-1)
                #print(Q_new.shape)
                S_list.append(Q_new - Q_old)
                Q_old = Q_new
                #print(np.array(S_list).shape)
                
                time_step += 1 # Loop counter
                vv1, vv2 = Qanderson, compare_v
                #np.max(Qanderson,axis = 0), np.max(compare_v, axis = 0)
                norm_diff = LA.norm((vv1-vv2).reshape(-1))
                self.normlist.append(norm_diff)
                
                '''
                if time_step % 100 == 0:
                    #reward = self.evaluate(251)
                    #self.returns.append(reward)
                    print('{} iteration,  residual : {:.4f}'.format(time_step, norm_diff))
                '''

                # Terminate condition - difference between updated vector
                if self.get_vbq(Qanderson.copy(), compare_v.copy(), self.precision):
                    break

            '''
            # Return logging for experiment 
            with open(file_name, "a") as myfile:
                myfile.write("## A3FIB reward ##")
                myfile.write("timestep\tavg.return\n")
                for idx, i in enumerate(self.returns):
                    myfile.write("%d\t%.4f\n"%(idx, i))
            '''

            alpha_end = time.time()
            self.time_step = time_step  # Total number of iterations for convergence
            self.total_time = alpha_end - alpha_start   # Total iteration time
            self.andy = andy
            self.naa = naa
    
    def update_RAA(self):
        print("Need implementation?")

    '''
    def chooseAction(self, cur_belief):
        return np.argmax(np.matmul(self.Q_value, cur_belief.T))
    ''' 
    def chooseAction(self, cur_belief):
        return np.argmax(np.matmul(self.rewards, cur_belief.T) + self.pomdp.discount * np.matmul(self.Q_value, cur_belief.T))


    def getValue(self, belief):
        value = np.max(np.matmul(self.Q_value, belief.T))
        return math.floor(value * 100) / 100.0

    def updateQValueFromValues(self):
        Q_value, Q = self.Q_value.copy(), self.Q_value.copy()
        # Tabular representation
        for a_index, action in enumerate(self.pomdp.actions):
            v = np.zeros([len(self.pomdp.states)])
            for o_index, obs in enumerate(self.pomdp.observations):
                # Need to convert this!
                #q = np.multiply(self.pomdp.O[(a_index, :, o_index)],self.pomdp.T[(a_index, :, :)])
                v += np.max(np.array([self.pomdp.OT[o_index][a_index].dot(Q[i,:]) for i in range(Q.shape[0])]),axis = 0)
            Q_value[a_index, :] = self.rewards[a_index, :]  + self.pomdp.discount * v
            
            # Do update
            #Q_value[a_index, :] = self.rewards[a_index, :] + self.pomdp.discount * v

        return Q_value, np.reshape(Q,-1)-np.reshape(Q_value,-1)

    def updateQValueFromSampleValues(self):
        Q_value, Q = self.Q_value.copy(), self.Q_value.copy()

        for a in range(self.a_num):
            v = np.zeros([self.s_num])
            for o in range(self.o_num):
                #v += np.max(np.array([O_hat.dot(Q[i,:]) for i in range(self.a_num)]))
                v += np.max(np.array([self.O_hat[o][a].dot(Q[i,:]) for i in range(self.a_num)]), axis = 0)          
            Q_value[a, :] = self.R_hat[a, :] + self.pomdp.discount * v

        self.Q_value = Q_value
        #Q_value = self.Q_value.copy()
        
        return Q_value, np.reshape(Q,-1)-np.reshape(Q_value,-1)

    def get_vbq(self,qv1,qv2, eps):
        # Caculate difference between update, to decide convergence
        delta = MIN
        v1, v2 = np.max(qv1,axis = 0), np.max(qv2,axis = 0)
        x = v1-v2
        delta = max(abs(x.min()), abs(x.max()))
        return delta < eps

class PERSEUS(OfflineSolver):
    def __init__(self, pomdp, precision = .001):
        
        """
        Initialization
        """        
        super(PERSEUS, self).__init__(pomdp, precision)
        self.Q_value = {}
        self.values = np.zeros([len(self.pomdp.states)])
        self.rewards = np.zeros([len(self.pomdp.actions), len(self.pomdp.states)])
        self.s_num, self.a_num, self.o_num = len(self.pomdp.states), len(self.pomdp.actions), len(self.pomdp.observations)
               
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

        reward_min, reward_max = np.min(self.rewards) , np.max(self.rewards)
        if reward_min == reward_max:
            if reward_max < 0:
                reward_max = 0
            else:
                reward_min = 0
        reward_min, reward_max = reward_min / (1-self.pomdp.discount), reward_max / (1-self.pomdp.discount) # Value function max/min
    
        max_abs_reward = np.max(np.abs(self.rewards))

    def chooseAction(self, cur_belief):
        # Q value as dictionary
        v_max, a_max = -np.inf, 0
        for a, _ in enumerate(self.pomdp.actions):
            try:
                self.Q_value[a]
            except:
                pass
            else:
                for _, q in enumerate(self.Q_value[a]):
                    #v = np.matmul(q, cur_belief.T)
                    v = np.matmul(self.rewards[a, :], cur_belief.T) + self.pomdp.discount * np.matmul(q, cur_belief.T)
                    if v > v_max:
                        v_max = v
                        a_max = a
        return a_max

    def getValue(self, belief):
        # Not in use
        value = np.max(np.matmul(self.Q_value, belief.T))
        return math.floor(value * 100) / 100.0
