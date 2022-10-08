import torch
import argparse
from utils import load_csv, kalman_filter
import statsmodels.discrete.discrete_model as sm
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Performance plots')
parser.add_argument('--recompute', action='store_true', default=False, help='recompute logprobs or plot')
args = parser.parse_args()

if args.recompute:
    regrets = torch.stack([torch.load('data/gershman_a2c' + str((float(i) + 1)*10) +  '.pth')[0] for i in range(1000)])
    rewards = torch.stack([torch.load('data/gershman_a2c' + str((float(i) + 1)*10) +  '.pth')[1] for i in range(1000)])
    actions = torch.stack([torch.load('data/gershman_a2c' + str((float(i) + 1)*10) +  '.pth')[2] for i in range(1000)])

    num_steps = rewards.shape[2]
    num_models = rewards.shape[0]
    num_episodes = rewards.shape[1]
    records = np.empty((num_models * num_steps * num_episodes, 8))

    counter = 0
    for model in tqdm(range(num_models)):
        for episode in range(num_episodes):
            for step in range(num_steps):
                records[counter, 0] = int(model) + 1
                records[counter, 1] = int(episode) + 1
                records[counter, 2] = int(step) + 1
                records[counter, 3] = 0
                records[counter, 4] = 0
                records[counter, 5] = actions[model, episode, step].item() + 1
                records[counter, 6] = rewards[model, episode, step].item() * 10
                records[counter, 7] = int(model)
                counter += 1

    fmt = ",".join(["%d"] + ["%d"] + ["%d"] + ["%.3f"] + ["%.3f"] + ["%d"] + ["%.3f"] + ["%.6f"])
    np.savetxt('data/simulations_all.csv', records, fmt=fmt, header="subject,block,trial,mu1,mu2,choice,reward,RT", comments='')

    Q = [10, 100, 100]
    current_data = load_csv('data/simulations_all.csv')
    latents_model = np.empty((1, len(current_data)), dtype=object)
    for s in range(len(current_data)):
        latents_model[0, s] = kalman_filter(Q, current_data[0, s])

    b_model = np.zeros((latents_model.shape[1], 3))
    for s in tqdm(range(latents_model.shape[1])):
        S = np.sqrt(latents_model[0, s]['s'][:, 0] + latents_model[0, s]['s'][:, 1])
        Sm = np.sqrt(latents_model[0, s]['s'][:, 0]) - np.sqrt(latents_model[0, s]['s'][:, 1])
        V = latents_model[0, s]['m'][:, 0] - latents_model[0, s]['m'][:, 1]
        C = (current_data[0, s]['c'] == 1)[:, 0].astype(float)
        X = np.stack([V, Sm, V/S]).T
        results = sm.Probit(C, X).fit()
        b_model[s, :] = results.params
    torch.save(b_model, 'data/regression_all.pth')
else:
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import uniform_filter1d
    from blr import BLR
    import matplotlib.colors as colors

    colors_list = list(colors._colors_full_map.values())

    b_model = torch.load('data/regression_all.pth')
    b_model[np.isnan(b_model)] = 0
    print(b_model.shape)
    print(b_model[0:10].mean(0))
    print(b_model[0:10].std(0))
    print(b_model[10:100].mean(0))
    print(b_model[10:100].std(0))
    print(b_model[100:1000].mean(0))
    print(b_model[100:1000].std(0))
    plt.rcParams["figure.figsize"] = (4.25,2.5)
    for i in range(3):
        linspace = torch.linspace(10, 10000, 1000)[:, None].float()
        model = BLR(1, 100, normalize=True, polynomials=7)
        model.fit(linspace, torch.from_numpy(b_model[:, [i]]).float())
        mean_pred, std_pred = model.predict(linspace)
        plt.plot(linspace, mean_pred, color='C' + str(i))

        plt.fill_between(
            linspace[:, 0],
            mean_pred[:, 0] - std_pred[:, 0],
            mean_pred[:, 0] + std_pred[:, 0],
            alpha=0.3,
            color='C' + str(i)
        )

    plt.legend(['Boltzmann', 'UCB', 'Thompson'], bbox_to_anchor=(0.0,1.02,1,0.2), loc="lower left", borderaxespad=0, ncol=3, frameon=False, handletextpad=0.5, mode='expand')
    plt.ylim(-0.5, 4.9)
    plt.xlabel('Description Length')
    plt.ylabel(r'$\mathbf{w}_i$')
    sns.despine()
    plt.tight_layout()
    plt.xlim(10, 10000)
    plt.xscale('log', base=10)
    plt.savefig('figures/gershman_exploration.pdf', bbox_inches='tight')
    plt.show()
