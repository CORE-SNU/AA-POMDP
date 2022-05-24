# -*- coding: utf-8 -*-
"""
Created on March 11 2019

@author: Wenjie Shi
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class raa(object):
    def __init__(self, num_critics, use_restart, reg=0.01):
        self.size = num_critics
        # regularization
        self.reg = reg
        self.use_restart = use_restart
        self.count = 0
        self.interval = 5000
        self.errors = torch.zeros(self.interval).to(device)
        self.opt_error = torch.tensor(0.).to(device)

    def calculate(self, Qs, F_Qs):
        Qs = Qs.squeeze(2).t()
        F_Qs = F_Qs.squeeze(2).t()

        delta_Qs = F_Qs - Qs
        cur_size = Qs.size(1)

        del_mat = delta_Qs.t().mm(delta_Qs)
        alpha = del_mat / torch.abs(torch.mean(del_mat))
        alpha += self.reg * torch.eye(cur_size).to(device)

        alpha = torch.sum(alpha.inverse(), 1)
        alpha = torch.unsqueeze(alpha / torch.sum(alpha), 1)

        # Local safeguarding - opt_gain, opt_obj    
        opt_obj = delta_Qs.mm(alpha)
        gk = delta_Qs[:, -1]
        opt_gain = opt_obj.norm() / gk.norm()

        # assert
        if self.use_restart:
            self.count += 1
            self.errors[self.count % self.interval] = torch.abs(torch.mean(delta_Qs[:, -1])).detach()

            if self.count % self.interval == 0:
                error = torch.mean(self.errors)
                if self.count == self.interval:
                    self.opt_error = error
                else:
                    self.opt_error = torch.min(self.opt_error, error)

                if (self.count > self.interval and error > self.opt_error) or self.count > 100000:
                    print(error, self.opt_error)
                    restart = True
                    self.count = 0
                else:
                    restart = False
            else:
                restart = False
        else:
            restart = False
    
        return alpha, restart, opt_gain, opt_obj.norm()

class a3(object):
    def __init__(self, num_critics, use_restart, reg=0.01):
        self.size = num_critics
        # regularization param
        self.reg = reg
        self.use_restart = use_restart
        self.count = 0
        self.interval = 5000
        self.errors = torch.zeros(self.interval).to(device)
        self.opt_error = torch.tensor(0.).to(device)

    def calculate(self, Qs, F_Qs):
        Qs = Qs.squeeze(2).t()
        F_Qs = F_Qs.squeeze(2).t()
        
        gs = Qs - F_Qs
        cur_size = Qs.size(1)
        gk = gs[:, -1].unsqueeze(1)
        Yk = gs.diff(dim=1)
        Sk = Qs.diff(dim=1)

        # Adaptive regularization - does this help in DRL setting?
        # eta = self.reg * ( Yk.norm(p='fro')  + Yk.norm(p='fro') ** 2 + Sk.norm(p='fro') ** 2 )
        eta = self.reg

        # Compute coeff
        reg_identity = eta * torch.eye(Yk.t().size(0)).to(device)
        inv = Yk.t().mm(Yk) + reg_identity
        gamma_ = inv.inverse().mm(Yk.t()).mm(gk)

        alpha = torch.zeros(cur_size, 1).to(device)
        alpha[0] = gamma_[0]
        alpha[1:-1] = gamma_.diff(axis=0)
        alpha[-1] = 1 - gamma_[-1]

        # Local safeguarding - opt_gain, opt_obj    
        opt_obj = gk - Yk.mm(gamma_)
        opt_gain = opt_obj.norm() / gk.norm()
        
        # Restart    
        if self.use_restart:
            self.count += 1
            self.errors[self.count % self.interval] = torch.abs(torch.mean(gs[:, -1])).detach()

            if self.count % self.interval == 0:
                error = torch.mean(self.errors)
                if self.count == self.interval:
                    self.opt_error = error
                else:
                    self.opt_error = torch.min(self.opt_error, error)

                if (self.count > self.interval and error > self.opt_error) or self.count > 100000:
                    print(error, self.opt_error)
                    restart = True
                    self.count = 0
                else:
                    restart = False
            else:
                restart = False
        else:
            restart = False
    
        return alpha, restart, opt_gain, opt_obj.norm()
