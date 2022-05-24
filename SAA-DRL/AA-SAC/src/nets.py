import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions.normal import Normal
import numpy as np


class SACActor(nn.Module):
    def __init__(self, dimS, dimA, hidden1, hidden2, ctrl_range):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, dimA)
        self.fc4 = nn.Linear(hidden2, dimA)

        self.ctrl_range = ctrl_range

    def forward(self, state, eval=False, with_log_prob=False):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.fc3(x)
        log_sigma = self.fc4(x)
        # clip value of log_sigma, as was done in Haarnoja's implementation of SAC:
        # https://github.com/haarnoja/sac.git
        log_sigma = torch.clamp(log_sigma, -20.0, 2.0)

        sigma = torch.exp(log_sigma)
        distribution = Independent(Normal(mu, sigma), 1)

        if not eval:
            # use rsample() instead of sample(), as sample() does not allow back-propagation through params
            u = distribution.rsample()
            if with_log_prob:
                log_prob = distribution.log_prob(u)
                log_prob -= 2.0 * torch.sum((np.log(2.0) + 0.5 * np.log(self.ctrl_range) - u - F.softplus(-2.0 * u)), dim=1)

            else:
                log_prob = None
        else:
            u = mu
            log_prob = None
        # apply tanh so that the resulting action lies in (-1, 1)^D
        a = self.ctrl_range * torch.tanh(u)

        return a, log_prob



class DoubleCritic(nn.Module):

    def __init__(self, dimS, dimA, hidden1, hidden2):
        super(DoubleCritic, self).__init__()
        self.fc1 = nn.Linear(dimS + dimA, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

        self.fc4 = nn.Linear(dimS + dimA, hidden1)
        self.fc5 = nn.Linear(hidden1, hidden2)
        self.fc6 = nn.Linear(hidden2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)

        return x1, x2

    def Q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x