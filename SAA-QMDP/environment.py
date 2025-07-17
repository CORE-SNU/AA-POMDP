"""
UW, CSEP 573, Win19
"""

import numpy as np
import random
from pomdp_accel import POMDP

class Environment:
    def __init__(self, pomdp):
        self.pomdp = pomdp
        r = random.uniform(0, 1)    # Probability r
        s = 0
        for i in range(len(pomdp.states)):
            s += pomdp.prior[i] # Accumulate belief
            # Accumulated belief over threshold
            # State must lie within belief!
            if s > r:
                # current state generated!
                self.cur_state = i
                break
            
    def act(self, action):
        """
        Perform the action
        return reward and observation
        reward = None means terminal state
        """
        #action, next state
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(self.pomdp.states)):
            try:
                s += self.pomdp.T[(action, self.cur_state, i)]
            except:
                pass
            if s > r:
                next_state = i
                break
        #observtion
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(self.pomdp.observations)):
            try:
                s += self.pomdp.O[(action, next_state, i)]
            except:
                pass
            if s > r:
                observation = i
                break
        # reward
        if self.pomdp.flag:
            try:
                reward = self.pomdp.R[(action, next_state)]
                #reward = self.pomdp.R[(action, self.cur_state)]
            except:
                reward = 0
        else:
            try:
                reward = self.pomdp.R[(action, self.cur_state)]
            except:
                reward = 0
        T = np.zeros([len(self.pomdp.actions)])
        for key, val in self.pomdp.T.items():
            a, s, s_ = key
            if s_ == next_state and s == next_state:
                T[a] = val
        '''
        if reward == 0 and np.where(T[:, next_state, next_state] < 1)[0].size == 0:
            reward = None
        '''
        if reward == 0 and np.where(T < 1)[0].size == 0:
            reward = None
        self.cur_state = next_state
        return reward, observation

            
    def _sample(self, cur_state, action):
        """
        Perform the action
        return reward and observation
        reward = None means terminal state
        """
        #action, next state
        r = random.uniform(0, 1)
        _s = 0
        next_state, observation, reward = 0, 0, 0
        for i in range(len(self.pomdp.states)):
            try:
                _s += self.pomdp.T[(action, cur_state, i)]
            except:
                pass
            if _s > r:
                next_state = i
                break
        #observtion
        r = random.uniform(0, 1)
        _s = 0
        for i in range(len(self.pomdp.observations)):
            try:
                _s += self.pomdp.O[(action, next_state, i)]
            except:
                pass
            if _s > r:
                observation = i
                break
        # reward
        if self.pomdp.flag:
            try:
                reward = self.pomdp.R[(action, next_state)]
                #reward = self.pomdp.R[(action, cur_state)]
            except:
                reward = 0
        else:
            try:
                reward = self.pomdp.R[(action, cur_state)]
            except:
                reward = 0
        T = np.zeros([len(self.pomdp.actions)])
        for key, val in self.pomdp.T.items():
            a, s, s_ = key
            if s_ == next_state and s == next_state:
                T[a] = val
        '''
        if reward == 0 and np.where(T[:, next_state, next_state] < 1)[0].size == 0:
            reward = None
        '''
        '''
        if reward == 0 and np.where(T < 1)[0].size == 0:
            reward = None
        '''
        return next_state, observation, reward

    # To sample parallely w.r.t (s, a) pairs
    def sample(self, arg):
        s, a, num_sample  = arg
        key, val = (s, a), []
        for i in range(num_sample):
            s_, o, r = self._sample(s,a)
            val.append((s_, o, r))

        return key, val
