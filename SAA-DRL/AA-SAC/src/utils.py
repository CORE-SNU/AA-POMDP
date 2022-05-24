import os
import torch
import gym
import numpy as np

def get_env_spec(env):
    print('environment : ' + env.unwrapped.spec.id)
    print('obs dim : ', env.observation_space.shape, '/ ctrl dim : ', env.action_space.shape)
    dimS = env.observation_space.shape[0]
    dimA = env.action_space.shape[0]
    ctrl_range = env.action_space.high[0]
    max_ep_len = env._max_episode_steps
    print('-' * 80)

    print('ctrl range : ({:.2f}, {:.2f})'.format(-ctrl_range, ctrl_range))
    print('max_ep_len : ', max_ep_len)
    print('-' * 80)

    return dimS, dimA, ctrl_range, max_ep_len


def set_log_dir(env_id):
    if not os.path.exists('./train_log/'):
        os.mkdir('./train_log/')
    if not os.path.exists('./eval_log/'):
        os.mkdir('./eval_log/')

    if not os.path.exists('./train_log/' + env_id + '/'):
        os.mkdir('./train_log/' + env_id + '/')
    if not os.path.exists('./eval_log/' + env_id + '/'):
        os.mkdir('./eval_log/' + env_id + '/')

    if not os.path.exists('./checkpoints/'):
        os.mkdir('./checkpoints')

    if not os.path.exists('./checkpoints/' + env_id + '/'):
        os.mkdir('./checkpoints/' + env_id + '/')
    return


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def render_agent(agent, env_id):
    eval_agent(agent, env_id, eval_num=1, render=True)


def eval_agent(agent, env_id, eval_num=5, render=False):
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


def load_model(agent, path, device='cpu'):
    print('networks loading...')
    checkpoint = torch.load(path)

    agent.pi.load_state_dict(checkpoint['actor'])
    agent.Q.load_state_dict(checkpoint['critic'])
    agent.target_Q.load_state_dict(checkpoint['target_critic'])
    agent.pi_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    agent.Q_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    return