#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
OfflinerSolver class with general functions
"""

#from pomdp_accel import POMDP
from environment import Environment
import numpy as np
from scipy.stats import entropy
from scipy.sparse import csr_matrix, dok_matrix, csc_matrix
from multiprocessing import Pool

class OfflineSolver:
    def __init__(self, pomdp, precision = .001):
        self.pomdp = pomdp
        self.precision = precision
        self.fixed_prior = self.pomdp.prior

    def _eval(self, max_iter = 251):
        """
        solve and calulcate the total reward for one run
        """
        total_reward = 0
        environment = Environment(self.pomdp)
        time_step = 0

        cur_belief = self.pomdp.prior

        # Rollout trajectory
        while time_step < max_iter:
            action = self.chooseAction(cur_belief)
            reward, obs = environment.act(action)
            if reward == None:  # we check Terminal states to get results faster
                break   # No terminal, to the best of my knowledge...
            total_reward += reward * (self.pomdp.discount ** time_step)
            cur_belief = self.updateBelief(cur_belief, action, obs)
            time_step +=1

        return total_reward

    def _solve(self, val):
        cur_belief, max_iter = val
        total_reward, time_step = 0, 0
        environment = Environment(self.pomdp)

        # Rollout trajectory
        while time_step < max_iter:
            action = self.chooseAction(cur_belief)
            reward, obs = environment.act(action)
            if reward == None:  # we check Terminal states to get results faster
                break   # No terminal, to the best of my knowledge...
            total_reward += reward * (self.pomdp.discount ** time_step)
            cur_belief = self.updateBelief(cur_belief, action, obs)
            time_step +=1

        return total_reward

    def evaluate(self, rand_init=False, num_runs=251):
        sum_reward, curr_reward = 0, 0
        reward_epi = np.zeros(num_runs)

        # To generate random prior each time solving pomdp
        # # of iterations - # of beliefs
        if rand_init:
            print('Randomized initial belief : True')
            rand_prior = np.random.random(len(self.pomdp.prior))
            rand_prior /= rand_prior.sum()
            self.pomdp.prior = rand_prior
        else:
            print('Randomized initial belief: False')
            self.pomdp.prior = self.fixed_prior

        for j in range(num_runs):
            curr_reward = self._eval(max_iter=num_runs)   # Get discounted return
            sum_reward += curr_reward
            reward_epi[j-1] = curr_reward
             
        # Length 251
        avg_reward = sum_reward / num_runs
        std_reward = np.std(reward_epi)
        print('Avg.Return : %.4f \t Std.Return : %.4f'%(avg_reward, std_reward))
        print('')

        return avg_reward, std_reward

    def _evaluate(self, rand_init=False, num_runs=251):
        sum_reward, curr_reward = 0, 0
        reward_epi = np.zeros(num_runs)
        
        b0 = []
        for j in range(num_runs):
            # To generate random prior each time solving pomdp
            # # of iterations - # of beliefs
            if rand_init:
                #print('Randomized initial belief : True')
                rand_prior = np.random.random(len(self.pomdp.prior))
                rand_prior /= rand_prior.sum()
                b0.append((rand_prior, num_runs))
            else:
                #print('Randomized initial belief: False')
                b0.append((self.fixed_prior, num_runs))
            #print('Initial prior entropy : %.3f bit'%(entropy(self.pomdp.prior, base=2)))
        
        pool = Pool(processes=4)
        result = pool.map(self._solve, b0)

        for idx, curr_reward in enumerate(result):
            sum_reward += curr_reward
            reward_epi[idx] = curr_reward

        avg_reward = sum_reward / num_runs
        std_reward = np.std(reward_epi)

        return avg_reward, std_reward

    def chooseAction(self, cur_belief):
        """
        Choose action (The best action based on the given belief)

        """
        raise NotImplementedError("Subclass must implement abstract method")

    def updateBelief(self, current_belief, action, observation):
        '''
        print(current_belief.shape)
        print(type(self.pomdp.t[action]))
        print(type(self.pomdp.o[action][observation]))
        (s,) (s, s) ()
        '''
        # Bayesian update
        current_belief = self.pomdp.t[action].dot(current_belief.T)
        current_belief = self.pomdp.o[action][observation] * current_belief

        new_belief = current_belief.T / np.sum(current_belief)

        return new_belief

    def getValue(self, cur_belief):
        """
        Return the estimated value function of the belief given as an input
        """
        raise NotImplementedError("Subclass must implement abstract method")
