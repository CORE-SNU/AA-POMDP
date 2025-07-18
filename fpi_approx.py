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
import click
from utils import *


class RandomPolicy(OfflineSolver):
    def __init__(self, pomdp, precision = .001, sample = False, initial_vector = None):
        
        """
        Initialization
        """        
        super(RandomPolicy, self).__init__(pomdp, precision)
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

    def chooseAction(self, cur_belief):
        return np.random.randint(self.a_num)


class MyopicPolicy(OfflineSolver):
    def __init__(self, pomdp, precision = .001, sample = False, initial_vector = None):
        
        """
        Initialization
        """        
        super(MyopicPolicy, self).__init__(pomdp, precision)
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

    def chooseAction(self, cur_belief):
        return np.argmax(np.matmul(self.rewards, cur_belief.T))   

@click.command()
@click.option('--env_name', default="sunysb")
@click.option('--alg_name', default='random')
#@click.option('--eval_per_trial', default=100)
#@click.option('--eval_length', default=100)
@click.option('--filename', default="./sunysb_random_policy.txt", help='location to store statistics')
def main(
        alg_name,
        env_name, 
        filename):

        print('')
        print('###########################')
        print('#     Starting Setups     #')
        print('###########################')
        pomdp = load_model(env_name)
        pomdp.print_summary()

        if alg_name == 'random':
            solver = RandomPolicy(pomdp)
                
            print('')
            print("alg : {} \t model : {}".format(alg_name, env_name))
            
            # Load saved model with designated param (algorithm name)
            stats = {'return_fix':[], 'return_rand':[]}
            for r in range(50):
                print('Evaluation process %d/%d'%(r+1, 50))
                # solver.Q_value = b[r]
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

        elif alg_name == 'myopic':
            solver = MyopicPolicy(pomdp)
                
            print('')
            print("alg : {} \t model : {}".format(alg_name, env_name))
            
            # Load saved model with designated param (algorithm name)
            stats = {'return_fix':[], 'return_rand':[]}
            for r in range(50):
                print('Evaluation process %d/%d'%(r+1, 50))
                # solver.Q_value = b[r]
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

    


if __name__ == '__main__':
    main()