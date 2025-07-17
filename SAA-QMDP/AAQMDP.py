'''
QMDP & A3QMDP Class
'''
from environment import Environment
from offlineSolver import OfflineSolver
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import time
import math
from multiprocessing import Pool
from scipy.sparse import csr_matrix, dok_matrix, csc_matrix, identity
from scipy.sparse import linalg as la
from scipy.special import logsumexp
import pickle

MIN = -np.inf

class AAQMDP(OfflineSolver):
    def __init__(self, 
        pomdp,
        QMDP,
        sample,
        max_iter, 
        eps,
        softmax_param,
        max_mem,
        eta,
        safeguard_coeff,
        max_type = "mellowmax",
        safeguard = "safe_local"
        ):

        """
        Initialization
        """
        super(QMDP, self).__init__(pomdp, eps)
        self.s_num = len(self.pomdp.states)
        self.a_num = len(self.pomdp.actions)
        self.o_num = len(self.pomdp.observations)
        self.QMDP = QMDP
        self.sample = sample
        self.max_iter = max_iter
        self.eps = eps
        self.softmax_param = softmax_param
        self.mem = max_mem
        self.eta = eta
        
        self.safeguard_coeff = safeguard_coeff
        self.safeguard = safeguard
        assert(
            self.safeguard == "safe_local"
            or self.safeguard == "safe_global"
            or self.safeguard == "strict_local"
            or self.safeguard == "opt_gain"
        )
        self.max_type = max_type
        assert(
            self.max_type == "standard"
            or self.max_type == "mellowmax"
            or self.max_type == "logsumexp"
        )

        
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

        reward_min, reward_max = np.min(self.rewards) , np.max(self.rewards)
        if reward_min == reward_max:
            if reward_max < 0:
                reward_max = 0
            else:
                reward_min = 0
        reward_min, reward_max = reward_min / (1-self.pomdp.discount), reward_max / (1-self.pomdp.discount) # Value function max/min
        self.Q_value = np.random.uniform(reward_min,reward_max,[len(self.pomdp.actions), len(self.pomdp.states)]) #if init else initial_vector
        self.total_time, self.andy, self.time_step, self.naa = 0, 0, 0, 0
        self.res, self.opt_gain, self.opt_res = [], [], []

        if self.QMDP:
            # Standard QMDP Loop
            time_step, start = 0, time.time()
            while time_step < self.max_iter:
                self.Q_value, gk = self.updateQValueFromValues()
                self.res.append(LA.norm(gk, np.inf))
                time_step += 1
                if self.res[-1] < self.eps:
                    break
            self.total_time = time.time() - start
            self.time_step = time_step
            self.andy = 0
            self.naa = 0
        else:
            self.solve_AA() # Go to AA loop
        

    def solve_AA(self):
        # Global safeguarding conditions
        phi, D = 1e-6, 1e6
        naa, Raa = 0, 0
        g0, safeguard = None, True
        R = self.max_iter / 10

        # Local safeguarding conditions
        m, m_bar = self.safeguard_coeff, 1
        opt_bar = self.safeguard_coeff
        local_speed = 1 # Non-expansive

        Qs, FQs, gs = [], [], []
        Ys, Ss = [], []

        start = time.time()
        andy = 0
        time_step = 0

        # First iteration
        FQ, gk = self.updateQValueFromValues()
        g0 = gk
        Qs.append(self.Q_value)
        FQs.append(FQ)
        gs.append(gk)
        self.Q_value = FQ
        self.res.append(LA.norm(gk, np.inf))
        time_step += 1
        
        while time_step < self.max_iter:
            """
            FPI step
            """
            FQ, gk = self.updateQValueFromValues()

            # Safeguarding 1
            if (self.safeguard == "strict_local"):
                if (LA.norm(gk, np.inf) > local_speed * LA.norm(gs[-1], np.inf)):
                    self.Q_value = FQs[-1]
                    FQ, gk = self.updateQValueFromValues()
                else:
                    naa +=1
            
            Qs.append(self.Q_value)
            FQs.append(FQ)
            gs.append(gk)
            Ys.append(gs[-1] - gs[-2])
            Ss.append(Qs[-1].reshape(-1) - Qs[-2].reshape(-1))
            self.res.append(LA.norm(gk, np.inf))
            
            # Stack memory & dropout old memory
            if len(Qs) > self.mem + 1:
                Qs.pop(0)
                FQs.pop(0)
                gs.pop(0)
                Ys.pop(0)
                Ss.pop(0)

            """
            AA weight calculation
            """
            and_start = time.time()

            # ck = np.sum( np.array([LA.norm(y, np.inf) for y in Ys]) / LA.norm(Ys[-1], np.inf) )
            Y = np.transpose(np.array(Ys))
            S = np.transpose(np.array(Ss))

            # Adaptive regularization
            eta = self.eta * ( LA.norm(np.array(Y), 'fro') + LA.norm(np.array(Y), 'fro')**2 + LA.norm(np.array(S), 'fro')**2 )
            delta = Y
            delta_t = np.transpose(delta)
            reg_identity = eta * np.identity(len(delta_t))
            delta_inverse = inv(np.add(np.matmul(delta_t,delta), reg_identity))
            gamma_ = np.matmul(np.matmul(delta_inverse, delta_t),gk)

            # Recover weight from gammas
            gamma_diff = np.diff(gamma_, n=1)            
            alpha_solution = np.concatenate(([gamma_[0]], gamma_diff, [1-gamma_[-1]]))
            if len(alpha_solution[1:-1]) > 2:
                Alpha_solution = np.concatenate((alpha_solution[1:-1], [gamma_[-2], 1]))
            else:
                Alpha_solution = alpha_solution
            andy += time.time() - and_start  # Elapsed time for AA weight caculation
            
            # qAA - [A X S], horizontal stack of value function vector
            Qanderson = np.zeros([len(self.pomdp.actions), len(self.pomdp.states)])
            gk_w = 0
            for cidx, coeff in enumerate(alpha_solution):
                Qanderson += coeff * FQs[cidx]
                gk_w += Alpha_solution[cidx] * gs[cidx]
            
            # For bound analysis!
            opt_gain = LA.norm(gs[-1] - np.matmul(Y, gamma_), np.inf) / LA.norm(gs[-1], np.inf)
            opt_gain_2 = LA.norm(gs[-1] - np.matmul(Y, gamma_)) / LA.norm(gs[-1])
            # opt_res_2 = LA.norm(gs[-1] - np.matmul(Y, gamma_)) ** 2 / LA.norm(gs[-1])
            opt_res_2 = LA.norm(gs[-1] - np.matmul(Y, gamma_)) ** 2
            coeff_rat = np.max(np.abs(gamma_)) / np.abs(gamma_[-1])**2
            self.opt_gain.append(opt_gain_2)
            self.opt_res.append(opt_res_2)
            # print(opt_gain_2, opt_res_2)
            
            if (self.safeguard == "opt_gain"):
                if opt_gain_2 > opt_bar:
                    Qanderson = FQs[-1]
                    Raa = 0 # number of AAs we used consecutively
                elif safeguard or Raa >= R:
                    if LA.norm(gk, np.inf) <= D*LA.norm(g0, np.inf)*(naa/R + 1)**(-(1+phi)):
                        naa += 1
                        Raa = 1
                        safeguard = False
                    else:
                        Qanderson = FQs[-1] 
                        Raa = 0
                else:
                    naa +=1
                    Raa +=1
            
            # Local safeguarding activated!
            if (self.safeguard == "safe_local"):
                approx_local_conv_2 = opt_gain_2 + m * opt_res_2

                if approx_local_conv_2 > m_bar:
                    Qanderson = FQs[-1]
                    Raa = 0 # number of AAs we used consecutively
                elif safeguard or Raa >= R:
                    if LA.norm(gk, np.inf) <= D*LA.norm(g0, np.inf)*(naa/R + 1)**(-(1+phi)):
                        naa += 1
                        Raa = 1
                        safeguard = False
                    else:
                        Qanderson = FQs[-1] 
                        Raa = 0
                else:
                    naa +=1
                    Raa +=1

            # Global safeguard
            if (self.safeguard == "safe_global"):
                if safeguard or Raa >= R:
                    if LA.norm(gk, np.inf) <= D*LA.norm(g0, np.inf)*(naa/R + 1)**(-(1+phi)):
                        naa += 1
                        Raa = 1
                        safeguard = False
                    else:
                        Qanderson = FQs[-1]  # Do not use AA update version
                        Raa = 0
                else:
                    naa +=1
                    Raa +=1

            self.Q_value = Qanderson
            time_step += 1 # Loop counter

            # Terminate condition - infinite norm!
            if self.res[-1] < self.eps:
                break

        self.time_step = time_step
        self.total_time = time.time() - start
        self.andy = andy
        self.naa = naa


    def update_RAA(self):
        print("Need implementation?")


    def chooseAction(self, cur_belief):
        return np.argmax(np.matmul(self.Q_value, cur_belief.T))


    def getValue(self, belief):
        value = np.max(np.matmul(self.Q_value, belief.T))
        return math.floor(value * 100) / 100.0


    def updateQValueFromValues(self):
        beta = 1 / self.softmax_param
        Q_value, Q = self.Q_value.copy(), self.Q_value.copy()

        # Hard max
        if self.max_type == "standard" and not self.sample:
            for a in range(self.a_num):
                v = np.zeros([self.s_num])
                for o in range(self.o_num):
                    v += np.max(np.array([self.pomdp.OT[o][a].dot(Q[i,:]) for i in range(Q.shape[0])]), axis = 0)
                Q_value[a, :] = self.rewards[a, :] + self.pomdp.discount * v
        
        if self.max_type == "standard" and self.sample:
            for a in range(self.a_num):
                v = np.zeros([self.s_num])
                for o in range(self.o_num):
                    v += np.max(np.array([self.O_hat[o][a].dot(Q[i,:]) for i in range(self.a_num)]), axis = 0)          
                Q_value[a, :] = self.R_hat[a, :] + self.pomdp.discount * v

        # logsumexp
        if self.max_type == "logsumexp" and not self.sample:
            for a in range(self.a_num):
                v = np.zeros([self.s_num])
                # logsumexp trick version
                for o in range(self.o_num):
                    q = np.array([self.pomdp.OT[o][a].dot(Q[i,:]) for i in range(Q.shape[0])])
                    trick = np.max(q) # As maximizing is faster than summation?
                    v += (1/beta) * np.log( np.sum( np.exp( beta * (q - trick), dtype=np.float128), axis = 0) ) + trick
                Q_value[a, :] = self.rewards[a, :] + self.pomdp.discount * v

        if self.max_type == "logsumexp" and self.sample:
            for a in range(self.a_num):
                v = np.zeros([self.s_num])
                for o in range(self.o_num):
                    q = np.array([self.pomdp.OT[o][a].dot(Q[i,:]) for i in range(Q.shape[0])])
                    trick = np.max(q) # As maximizing is faster than summation?
                    v += (1/beta) * np.log( np.sum( np.exp( beta * (q - trick), dtype=np.float128), axis = 0) ) + trick
                Q_value[a, :] = self.R_hat[a, :] + self.pomdp.discount * v

        # mellowmax
        if self.max_type == "mellowmax" and not self.sample:
            for a in range(self.a_num):
                v = np.zeros([self.s_num])
                for o in range(self.o_num):
                    q = np.array([self.pomdp.OT[o][a].dot(Q[i,:]) for i in range(Q.shape[0])])
                    trick = np.max(q)
                    v += (1/beta) * np.log( np.sum( np.exp( beta * (q - trick) , dtype=np.float128), axis = 0) / self.a_num ) + trick
                Q_value[a, :] = self.rewards[a, :] + self.pomdp.discount * v

        if self.max_type == "mellowmax" and self.sample:
            for a in range(self.a_num):
                v = np.zeros([self.s_num])
                for o in range(self.o_num):
                    q = np.array([self.O_hat[o][a].dot(Q[i,:]) for i in range(Q.shape[0])])
                    trick = np.max(q)
                    v += (1/beta) * np.log( np.sum( np.exp( beta * (q - trick) , dtype=np.float128), axis = 0) / self.a_num ) + trick
                Q_value[a, :] = self.R_hat[a, :] + self.pomdp.discount * v

        return Q_value, np.reshape(Q,-1) - np.reshape(Q_value,-1)


class SARSOP(OfflineSolver):
    def __init__(self, pomdp, precision = .001):
        
        """
        Initialization
        """        
        super(SARSOP, self).__init__(pomdp, precision)
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
