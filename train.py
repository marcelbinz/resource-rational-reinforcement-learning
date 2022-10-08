import gym
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import envs.bandits
from rl3 import RL3A2C
import argparse
from tqdm import tqdm

def run(env_name, c, gamma, prior, num_episodes, print_every, save_every, num_hidden, save_dir, device, lr, reward_scaling, reward_shifting, unconstrained, batch_size=32):

    writer = SummaryWriter('runs/' + save_dir)

    env = gym.make(env_name)
    env.batch_size = batch_size
    env.reward_scaling = reward_scaling
    env.reward_shifting = reward_shifting
    env.device = device

    model = RL3A2C(env.observation_space.low.shape[0], env.action_space.n, num_hidden, prior, bias=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for t in tqdm(range(int(num_episodes))):
        regret = 0
        done = False
        obs = env.reset()

        hx = model.initial_states(batch_size)
        zeta = model.get_zeta(batch_size)

        # if the task has forced choice trials
        if obs.dim() == 3:
            for i in range(obs.shape[0] - 1):
                _, _, hx = model(obs[i], hx, zeta)
            obs = obs[obs.shape[0] - 1].clone()

        rewards = []
        values = []
        log_probs = []

        while not done:
            policy, value, hx, action = model.act(obs, hx, zeta)
            values.append(value)
            log_probs.append(policy.log_prob(action))

            obs, reward, done, info = env.step(action)
            regret += info['regrets']
            rewards.append(reward)

        values.append(torch.zeros_like(values[-1]))

        loss = 0
        rewards = torch.stack(rewards, axis=1)
        for i in range(rewards.shape[1]):
            advantage = rewards[:, i] + gamma * values[i + 1].squeeze().detach() - values[i].squeeze()
            loss = loss + advantage.pow(2).mean() - (log_probs[i] * advantage.detach()).mean()

        optimizer.zero_grad()
        kld = model.kl_divergence()

        # gradient step
        if unconstrained:
            loss = (loss / env.max_steps)
            loss.backward()
        else:
            # 1. model parameters loss
            loss = torch.exp(model.beta.detach()) * kld + (loss / env.max_steps)
            # 2. beta loss
            loss_beta = -torch.exp(model.beta) * (kld - c).detach()
            (loss + loss_beta).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optimizer.step()

        if (not t % print_every):
            writer.add_scalar('Rewards', rewards.sum(-1).mean(), t)
            writer.add_scalar('Regret', regret, t)
            writer.add_scalar('Loss', loss, t)
            writer.add_scalar('KLD', kld, t)
            writer.add_scalar('beta', torch.exp(model.beta), t)

        if (not t % save_every):
            torch.save([c, t, model], save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--num-episodes', type=int, default=1e6, help='number of trajectories for training')
    parser.add_argument('--print-every', type=int, default=100, help='how often to print')
    parser.add_argument('--save-every', type=int, default=100, help='how often to save')
    parser.add_argument('--runs', type=int, default=1, help='total number of runs')
    parser.add_argument('--first-run-id', type=int, default=0, help='id of the first run')

    parser.add_argument('--num_hidden', type=int, default=128, help='number of hidden units')
    parser.add_argument('--c', type=float, default=500, help='number of assumed samples')
    parser.add_argument('--offset', type=float, default=0, help='kl constraint offset')
    parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
    parser.add_argument('--reward-scaling', type=float, default=1.0, help='reward scaling')
    parser.add_argument('--reward-shifting', type=float, default=0.0, help='reward shifting')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--prior', default='gaussian', help='type of prior')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--unconstrained', action='store_true', default=False, help='no KL term')
    parser.add_argument('--env-name', default='gershman2018deconstructing-v0', help='name of the environment')
    parser.add_argument('--save-dir', default='trained_models/', help='directory to save models')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.env_name == 'bechara1994training-v0':
        args.c = args.c * 10
    if args.env_name == 'gershman2018deconstructing-v0':
        args.c = (args.c * 10) - args.offset
    if args.env_name == 'somerville2017charting-v2':
        args.c = args.c * 100
        
    for i in range(args.runs):
        if args.unconstrained:
            save_dir = args.save_dir + 'env=' + args.env_name  + '_prior=' + args.prior + '_constraint=unconstrained_algo=a2c_run=' + str(args.first_run_id + i) + '.pt'
        else:
            save_dir = args.save_dir + 'env=' + args.env_name  + '_prior=' + args.prior + '_constraint=' + str(args.c) + '_algo=a2c_run=' + str(args.first_run_id + i) + '.pt'
        run(args.env_name, args.c, args.gamma, args.prior, args.num_episodes, args.print_every, args.save_every, args.num_hidden, save_dir, device, args.lr, args.reward_scaling, args.reward_shifting, args.unconstrained)
