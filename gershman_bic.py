from utils import load_csv, kalman_filter
import numpy as np
import math
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.discrete.discrete_model as sm
from groupBMC.groupBMC import GroupBMC

plt.rcParams["figure.figsize"] = (2.5,2.)

current_data = load_csv('data/data2.csv')
latents_human = np.empty((1, len(current_data)), dtype=object)

for s in range(len(current_data)):
    latents_human[0, s] = kalman_filter([1, 100, 100], current_data[0, s])
    print(latents_human[0, s]['s'])

titles = ['Subject ' + str(i) for i in range(latents_human.shape[1])]

bics = np.zeros(latents_human.shape[1])
for s in range(latents_human.shape[1]):
    S = np.sqrt(latents_human[0, s]['s'][:, 0] + latents_human[0, s]['s'][:, 1])
    Sm = np.sqrt(latents_human[0, s]['s'][:, 0]) - np.sqrt(latents_human[0, s]['s'][:, 1])
    V = latents_human[0, s]['m'][:, 0] - latents_human[0, s]['m'][:, 1]
    C = (current_data[0, s]['c'] == 1)[:, 0].astype(float)
    X = np.stack([V, Sm, V/S]).T

    results = sm.Probit(C, X).fit()
    bics[s] = results.llf - (0.5 * 3 * math.log(X.shape[0]))

bics_value = np.zeros(latents_human.shape[1])
for s in range(latents_human.shape[1]):
    S = np.sqrt(latents_human[0, s]['s'][:, 0] + latents_human[0, s]['s'][:, 1])
    Sm = np.sqrt(latents_human[0, s]['s'][:, 0]) - np.sqrt(latents_human[0, s]['s'][:, 1])
    V = latents_human[0, s]['m'][:, 0] - latents_human[0, s]['m'][:, 1]
    C = (current_data[0, s]['c'] == 1)[:, 0].astype(float)
    X = V[:, None]

    results = sm.Probit(C, X).fit()
    bics_value[s] = results.llf - (0.5 * 1 * math.log(X.shape[0]))

bics_ucb = np.zeros(latents_human.shape[1])
for s in range(latents_human.shape[1]):
    S = np.sqrt(latents_human[0, s]['s'][:, 0] + latents_human[0, s]['s'][:, 1])
    Sm = np.sqrt(latents_human[0, s]['s'][:, 0]) - np.sqrt(latents_human[0, s]['s'][:, 1])
    V = latents_human[0, s]['m'][:, 0] - latents_human[0, s]['m'][:, 1]
    C = (current_data[0, s]['c'] == 1)[:, 0].astype(float)
    X = np.stack([V, Sm]).T

    results = sm.Probit(C, X).fit()
    bics_ucb[s] = results.llf - (0.5 * 2 * math.log(X.shape[0]))

bics_ts = np.zeros(latents_human.shape[1])
for s in range(latents_human.shape[1]):
    S = np.sqrt(latents_human[0, s]['s'][:, 0] + latents_human[0, s]['s'][:, 1])
    Sm = np.sqrt(latents_human[0, s]['s'][:, 0]) - np.sqrt(latents_human[0, s]['s'][:, 1])
    V = latents_human[0, s]['m'][:, 0] - latents_human[0, s]['m'][:, 1]
    C = (current_data[0, s]['c'] == 1)[:, 0].astype(float)
    X = (V/S)[:, None]

    results = sm.Probit(C, X).fit()
    bics_ts[s] = results.llf - (0.5 * 1 * math.log(X.shape[0]))

likelihoods = torch.stack([torch.load('data/gershman_subject=' + str(i) +  '_prior=svdo.pth') for i in range(44)])
likelihoods = likelihoods[:, -1, :]
a, b = likelihoods.max(-1)
bic_rl2_new = a.numpy() - (0.5 * 1 * math.log(200))

likelihoods = torch.stack([torch.load('data/gershman_subject=' + str(i) +  '_prior=svdo.pth') for i in range(44)])
print('here')
print(likelihoods.shape)
likelihoods = likelihoods.reshape(44, -1)
a, b = likelihoods.max(-1)
bic_rl3_new = a.numpy() - (0.5 * 2 * math.log(200))
print(bic_rl3_new.shape)

guessing = (torch.log(torch.ones(44) * 0.5) * 200).numpy()

L = np.array([bic_rl3_new, bics, bics_value, bics_ucb, bics_ts, bic_rl2_new])
torch.save(L, 'data/all_models_likelihoods.pth')
print(L.shape)
print((L.argmax(0) == 0).sum())
print(-2*L.sum(1))
result = GroupBMC(L).get_result()

print(result.frequency_mean)
print(result.exceedance_probability)
print(result.protected_exceedance_probability)

plt.bar(np.arange(L.shape[0]), 2 * (-L.sum(1)), alpha=0.8)
plt.xticks(np.arange(L.shape[0]), [r'RR-RL$^2$', r'Hybrid', 'Boltzmann', 'UCB', 'Thompson', r'RL$^2$'], rotation=90)
plt.ylim(5500)
plt.ylabel(r'BIC')
sns.despine()
plt.savefig('figures/bic.pdf', bbox_inches='tight')
plt.show()
