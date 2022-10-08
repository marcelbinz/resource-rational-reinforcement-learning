import torch
import gym
import envs.bandits
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--recompute', action='store_true', default=False, help='recompute logprobs or plot')
parser.add_argument('--subject', type=int, default=0, help='subject id')
parser.add_argument('--reward-scaling', type=float, default=15.0, help='reward scaling')
parser.add_argument('--num-episodes', type=int, default=10000, help='number of episodes')
parser.add_argument('--runs', type=int, default=1, help='total number of runs')
parser.add_argument('--first-run-id', type=int, default=0, help='id of the first run')
args = parser.parse_args()

if args.recompute:
    save_path = 'data/bechara_a2c' + str(args.subject) + '.pth'
    num_runs = args.runs - args.first_run_id
    actions = torch.zeros(num_runs, args.num_episodes, 4)
    regrets = torch.zeros(num_runs, args.num_episodes, 100)

    for run in range(args.first_run_id, args.first_run_id + args.runs):
        file_name = 'trained_models/env=bechara1994training-v0_prior=svdo_constraint=' + str(float(args.subject)) + 'algo=a2c_run=' + str(run) + '.pt'

        env = gym.make('bechara1994insensitivity-v0')
        env.batch_size = 1
        env.reward_scaling = args.reward_scaling
        env.device = 'cpu'

        _, _, agent = torch.load(file_name, map_location='cpu')
        for t in tqdm(range(args.num_episodes)):
            done = False
            obs = env.reset()

            hx = agent.initial_states(env.batch_size)
            zeta = agent.get_zeta(env.batch_size)

            while not done:
                if args.a2c:
                    _, _, hx, action = agent.act(obs, hx, zeta)
                else:
                    _, hx, action = agent.act(obs, hx, zeta)
                obs, reward, done, info = env.step(action)

                actions[run, t, action.item()] += 1
                regrets[run, t, env.t-1] = info['regrets']

    torch.save([actions, regrets], save_path)
else:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from blr import BLR

    linspace = np.linspace(100, 10000, 100)
    actions = torch.stack([torch.load('data/bechara_a2c' + str((i + 1)*100) +  '.pth')[0] for i in range(100)])
    regrets = torch.stack([torch.load('data/bechara_a2c' + str((i + 1)*100) +  '.pth')[1] for i in range(100)])
    file_names = ['trained_models/env=bechara1994training-v0_prior=svdo_constraint=' + str(beta) + 'algo=a2c_run=0.pt' for beta in linspace]

    print(actions.shape)
    plt.rcParams["figure.figsize"] = (6,3)

    actions = actions.mean(1).mean(1)
    regrets = regrets.mean(1)

    klds = []
    stable_regrets = []
    actions_low_risk = []
    actions_high_risk = []

    for i, file_name in enumerate(file_names):
        stable_regrets.append(regrets[i])
        actions_low_risk.append(actions[i,2] + actions[i,3])
        actions_high_risk.append(actions[i,0] + actions[i,1])

    y1 = torch.Tensor(actions_low_risk)[:, None].float() / 100
    y2 = torch.Tensor(actions_high_risk)[:, None].float() / 100
    x = torch.from_numpy(linspace[:, None]).float()


    model = BLR(1, 100, normalize=True, polynomials=3)
    model.fit(x, y1)
    mean_pred_low, std_pred_low = model.predict(x)

    model = BLR(1, 100, normalize=True, polynomials=3)
    model.fit(x, y2)
    mean_pred_high, std_pred_high = model.predict(x)

    stable_regrets = torch.stack(stable_regrets)

    plt.rcParams["figure.figsize"] = (3,3)
    x = np.arange(2)
    width = 0.35

    plt.bar(x - width/2, [mean_pred_low[-1,0], mean_pred_low[19,0]], width, label='low-risk', alpha=0.8)
    plt.bar(x + width/2, [mean_pred_high[-1,0], mean_pred_high[19,0]], width, label='high-risk', alpha=0.8)
    sns.despine()
    plt.legend(bbox_to_anchor=(0.0,1.02,1,0.2), loc="lower left", ncol=3, frameon=False, handletextpad=0.5, mode='expand')
    plt.tight_layout()
    plt.ylim(0, 1.15)
    plt.ylabel('Choice Probability')
    plt.xticks(np.arange(2), [r'KL$=10000$', r'KL$=2000$'])
    plt.savefig('figures/bechara_model.pdf', bbox_inches='tight')
    plt.show()
