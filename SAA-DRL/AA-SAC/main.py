import numpy as np
import torch
import gym
import argparse
import os

import src.replay as replay
from src.sac import SACAgent
from src.utils import get_env_spec, set_log_dir

# For logging!
import pandas as pd
import wandb

def eval_agent(agent, env_id, eval_num=10, render=False):
    log = []
    for ep in range(eval_num):
        env = gym.make(env_id)

        state = env.reset()
        step_count = 0
        ep_reward = 0
        done = False

        while not done:
            if render and ep == 0:
                env.render()

            action = agent.act(state, eval=True)
            next_state, reward, done, _ = env.step(action)
            step_count += 1
            state = next_state
            ep_reward += reward

        if render and ep == 0:
            env.close()
        log.append(ep_reward)

    avg = sum(log) / eval_num

    return avg



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", default="A3-SAC")              # Policy name
    parser.add_argument("--env_name", default="Hopper-v3")            # OpenAI gym environment name Pendulum-v0
    parser.add_argument("--seed", default=101, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)     # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=1e4, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_steps", default=1e6, type=float)         # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")           # Whether or not models are saved
    parser.add_argument("--batch_size", default=256, type=int)          # Batch size for both actor and critic
    parser.add_argument("--episode_size", default=1000, type=int)       # Maximum time step in an episode
    parser.add_argument("--discount", default=0.99, type=float)         # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)             # Target network update rate,default: 0.005
    parser.add_argument('--train_interval', required=False, default=50, type=int) # Adopted from https://github.com/CORE-SNU/drlcourse/blob/main/day4/sac/sac.py
    parser.add_argument("--alpha", default=0.2, type=float)        # Range to clip target policy noise
    parser.add_argument("--lr", default=1e-3, type=float)        # Range to clip target policy noise
    # for regularized anderson acceleration
    parser.add_argument("--use_restart", action="store_true")           # Whether to use the restart technique
    parser.add_argument("--beta", default=0.0, type=float)              # Coefficient of consistency between actor and critic
    parser.add_argument("--reg_scale", default=10, type=float)       # Scale of regularization for anderson acceleration
    parser.add_argument("--num", default=5, type=int)                   # Maximal number for previous critics
    parser.add_argument("--aa_batch", default=256, type=int)            # The batch size for calculating the weights in AA
    parser.add_argument("--theta_thres", default=1.5, type=float)   # Local safeguarding coeffcient for AA
    # Implement safeguard_freq?
    parser.add_argument("--safeguard_freq", default=1000, type=int)
    # gpu setup
    parser.add_argument("--gpu", type=int, default=0, help="ID of GPU to be used")

    args = parser.parse_args()
    data_path = "./logs/{}/{}-num={}/seed-{}/data".format(args.env_name, args.policy_name, args.num, args.seed)
    model_path = "./logs/{}/{}-num={}/seed-{}/model".format(args.env_name, args.policy_name, args.num, args.seed)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if args.save_models and not os.path.exists(model_path):
        os.makedirs(model_path)

    # command
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_iter = int(args.max_steps)
    env = gym.make(args.env_name)
    
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dimS, dimA, ctrl_range, max_ep_len = get_env_spec(env)

    max_ep_len = args.episode_size

    # Initialize policy
    if args.agent_name == "SAC": aa_type=None
    elif args.agent_name == "A3-SAC": aa_type="A3"
    elif args.agent_name == "RAA-SAC": aa_type="RAA"
    else: raise NotImplementedError

    agent = SACAgent(
                     dimS,
                     dimA,
                     ctrl_range,
                     gamma=args.discount,
                     pi_lr=args.lr,
                     q_lr=args.lr,
                     polyak=args.tau,
                     alpha=args.alpha,
                     hidden1=256,
                     hidden2=256,
                     buffer_size=1e6,
                     batch_size=args.batch_size,
                     device=device,
                     aa_type=aa_type,
                     use_restart=args.use_restart,
                     beta=args.beta, 
                     reg_scale=args.reg_scale, 
                     num=args.num, 
                     aa_batch=args.aa_batch, 
                     theta_thres=args.theta_thres,
                     safeguard_freq=args.safeguard_freq
                     )

    # set_log_dir(args.env_name)

    obs = env.reset()
    step_count = 0
    ep_reward = 0
    ep_count = 0
    eval_num = 10
    # Logger
    log = {
        'residuals':[],
        'evaluations':[],
        'opt_gain':[],
        'opt_obj':[]
    }

    # main loop
    residual, opt_gain, opt_obj = 0, 0, 0
    for t in range(max_iter + 1):
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)

        next_obs, reward, done, _ = env.step(action)
        step_count += 1

        if step_count == max_ep_len:
            done = False

        agent.buffer.append(obs, action, next_obs, reward, done)

        obs = next_obs
        ep_reward += reward

        if done or (step_count == max_ep_len):
            ep_count += 1
            # train_logger.writerow([t, ep_reward])
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(ep_count, t, step_count, round(ep_reward, 2)))
            obs = env.reset()
            step_count = 0
            ep_reward = 0

        if (t >= args.start_timesteps) and (t % args.train_interval == 0):
            for _ in range(args.train_interval):
                residual, opt_gain, opt_obj = agent.train()

        if t % args.eval_freq == 0:
            eval_score = eval_agent(agent, args.env_name, render=False)
            # print('step {} : {:.4f}'.format(t, eval_score))
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(eval_num, round(eval_score, 2)))
            print("----------------------------------------")
            
            log["residuals"].append(residual)
            log["evaluations"].append(eval_score)
            log["opt_gain"].append(opt_gain)
            log["opt_obj"].append(opt_obj)
    
    df = pd.DataFrame(log)
    df.to_csv(data_path + '.csv', index=False)
    
    


def render_agent(agent, env_id):
    eval_agent(agent, env_id, eval_num=1, render=True)
