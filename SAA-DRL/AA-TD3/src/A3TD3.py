import torch
import threading
from torch.autograd import Variable
import torch.nn.functional as F
from src.nets import Actor, Critic
from src.anderson import a3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Adaptive Anderson Acceleration based Twin Delayed Deep Deterministic Policy Gradients (A3TD3)

class A3TD3(object):
    def __init__(self, state_dim, action_dim, max_action, use_restart=True,
                 beta=0.1, reg_scale=0.01, num=5, aa_batch=500, 
                 theta_thres=0.99, safeguard_freq=2000):
        self.max_action = max_action

        self.num = num
        self.beta = beta
        self.aa_batch = aa_batch
        self.anderson = a3(num, use_restart, reg_scale)
        self.interval = safeguard_freq
        self.restart = True
        self.cur_num = 1
        self.theta_thres = theta_thres # Parameter for local convergence rate control
        # self.theta_min = theta_min

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critics = self.anderson_critics(state_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def anderson_critics(self, state_dim, action_dim):
        target_critics = []
        for _ in range(self.num):
            new_critic = Critic(state_dim, action_dim).to(device)
            new_critic.load_state_dict(self.critic.state_dict())
            target_critics.append(new_critic)
        return target_critics

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def Bellman_residual(self, state, action, reward, done, next_state, discount):
        Qs, _ = self.critic(state, action)
        next_Qs, _ = self.critic(next_state, self.actor(next_state))
        residual = torch.mean(torch.abs(reward + done * discount * next_Qs - Qs)) # Maybe used for logging...?
        return residual.cpu().detach().numpy()

    def critic_forward(self, i, num, state, action, catQs):
        catQ1, catQ2 = self.target_critics[i-num](state, action)
        catQs[i] = torch.min(catQ1, catQ2)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        residual, opt_gain, opt_obj = 0, 0, 0

        for it in range(iterations):

            # Sample replay buffer
            sample_size = min(replay_buffer.length(), self.aa_batch)
            train_batch = min(sample_size, batch_size) # Why different train batch and sample batch?
            x, y, u, r, d = replay_buffer.sample(sample_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            # For the training of actor
            a_action = self.actor(state[:train_batch, :])

            # Compute the target Q value
            if self.restart:
            # or it % policy_freq == 0: # restart is same mechanism as safeguarding!
                self.cur_num = 1
                self.restart = False
                next_Q1, next_Q2 = self.target_critics[-1](next_state[:train_batch, :], next_action[:train_batch, :])
                next_Q = torch.min(next_Q1, next_Q2)
                target_Q = reward[:train_batch, :] + (done[:train_batch, :] * discount * next_Q).detach()
            else: # use AA?
                self.cur_num += 1
                num = min(self.num, self.cur_num)

                if it % policy_freq == 0:
                    cat_state = torch.cat((state, next_state, state[:train_batch, :]), 0)
                    cat_action = torch.cat((action, next_action, a_action), 0)
                else:
                    cat_state = torch.cat((state, next_state), 0)
                    cat_action = torch.cat((action, next_action), 0)

                catQs = []
                for i in range(num, 0, -1): # reversed
                    catQ1, catQ2 = self.target_critics[-i](cat_state, cat_action) # k-num ... k
                    catQs.append(torch.min(catQ1, catQ2)) # Double critic
                catQs = torch.cat(catQs, 0).view(num, -1, 1)

                # sampled AA? - consideration for speed, maybe...
                Qs, next_Qs = catQs[:, :sample_size, :], catQs[:, sample_size:2*sample_size, :]
                F_Qs = torch.cat([(reward + done * discount * Q).unsqueeze(0) for Q in next_Qs], 0)

                alpha, restart, opt_gain, opt_obj = self.anderson.calculate(Qs, F_Qs)
                self.restart = restart # Keep restart or not?
                
                # Safeguarding here! - need to parameterize this
                if opt_gain > self.theta_thres and it % self.interval == 0:
                    alpha = 0*alpha
                    alpha[-1] = 1 # Just a fixed-point iteration
                else:
                    print("Optimization gain : {:2f}".format(opt_gain))
                
                target_Qs = self.beta * Qs[:, :train_batch, :] + (1 - self.beta) * F_Qs[:, :train_batch, :]
                target_Qs = target_Qs.squeeze(2).t()
                target_Q = (target_Qs.mm(alpha)).detach() # Matrix multiplication

                # for policy
                if it % policy_freq == 0:
                    policy_Q = catQs[:, -train_batch:, :].squeeze(2).t().mm(alpha.detach())

            # Compute critic loss
            current_Q1, current_Q2 = self.critic(state[:train_batch, :], action[:train_batch, :])
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            """
            Original version from RAA-DRL
            """
            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                biased_loss = -self.critic.Q1(state[:train_batch, :], a_action).mean()
                if self.cur_num == 1:
                    actor_loss = biased_loss
                else:
                    true_loss = -policy_Q.mean()
                    actor_loss = self.beta * true_loss + (1-self.beta) * biased_loss # Why beta here...?

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                self.target_update(tau)

            # Compute the Bellman residual
            if it == iterations-1:
                residual = self.Bellman_residual(state, action, reward, done, next_state, discount)

        return residual, opt_gain.cpu().detach().numpy(), opt_obj.cpu().detach().numpy()

    def target_update(self, tau):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, last_target_param, target_param in zip(self.critic.parameters(),
                                                          self.target_critics[-1].parameters(),
                                                          self.target_critics[0].parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * last_target_param.data)
        self.target_critics.append(self.target_critics[0])
        self.target_critics.remove(self.target_critics[0])

    def save(self, directory):
        torch.save(self.actor.state_dict(), '%s/actor.pth' % (directory))
        torch.save(self.critic.state_dict(), '%s/critic.pth' % (directory))

    def load(self, directory):
        self.actor.load_state_dict(torch.load('%s/actor.pth' % (directory)))
        self.critic.load_state_dict(torch.load('%s/critic.pth' % (directory)))
