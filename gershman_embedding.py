import torch
import gym
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
import envs.bandits
from tqdm import tqdm
import argparse
from utils import load_csv, kalman_filter
import statsmodels.discrete.discrete_model as sm
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import umap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from pylab import rcParams
import matplotlib as mpl

Q = [10, 100, 100]

np.set_printoptions(precision=2, suppress=True)

current_data = load_csv('data/data2.csv')
latents_human = np.empty((1, len(current_data)), dtype=object)

for s in range(len(current_data)):
    latents_human[0, s] = kalman_filter(Q, current_data[0, s])

titles = ['Subject ' + str(i) for i in range(latents_human.shape[1])]

b_human = np.zeros((latents_human.shape[1], 3))
human_stds = np.zeros((latents_human.shape[1], 3))
for s in range(latents_human.shape[1]):
    S = np.sqrt(latents_human[0, s]['s'][:, 0] + latents_human[0, s]['s'][:, 1])
    Sm = np.sqrt(latents_human[0, s]['s'][:, 0]) - np.sqrt(latents_human[0, s]['s'][:, 1])
    V = latents_human[0, s]['m'][:, 0] - latents_human[0, s]['m'][:, 1]
    C = (current_data[0, s]['c'] == 1)[:, 0].astype(float)
    X = np.stack([V, Sm, V/S]).T
    results = sm.Probit(C, X).fit()
    b_human[s, :] = results.params

trans = umap.UMAP().fit(b_human)

likelihoods = torch.stack([torch.load('data/gershman_subject=' + str(i) +  '_prior=svdo.pth') for i in range(44)])
likelihoods = likelihoods[:, :1000, :]
likelihoods, _ = likelihoods.max(-1)
_, best_index = likelihoods.max(-1)
print(best_index)

regrets = torch.stack([torch.load('data/gershman_a2c' + str(float(i) + 1) +  '.pth')[0] for i in range(1000)])
rewards = torch.stack([torch.load('data/gershman_a2c' + str(float(i) + 1) +  '.pth')[1] for i in range(1000)])
actions = torch.stack([torch.load('data/gershman_a2c' + str(float(i) + 1) +  '.pth')[2] for i in range(1000)])

num_steps = rewards.shape[2]
num_models = 44
num_episodes = 1000
records = np.empty((num_models * num_steps * num_episodes, 8))

counter = 0
for model in range(num_models):
    for episode in range(num_episodes):
        for step in range(num_steps):
            records[counter, 0] = int(model) + 1
            records[counter, 1] = int(episode) + 1
            records[counter, 2] = int(step) + 1
            records[counter, 3] = 0
            records[counter, 4] = 0
            records[counter, 5] = actions[best_index[model], episode, step].item() + 1
            records[counter, 6] = rewards[best_index[model], episode, step].item() * 10
            records[counter, 7] = int(best_index[model])
            counter += 1

fmt = ",".join(["%d"] + ["%d"] + ["%d"] + ["%.3f"] + ["%.3f"] + ["%d"] + ["%.3f"] + ["%.6f"])
np.savetxt('data/fitted_simulations.csv', records, fmt=fmt, header="subject,block,trial,mu1,mu2,choice,reward,RT", comments='')

current_data = load_csv('data/fitted_simulations.csv')
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

b = np.concatenate((b_human, b_model))
print(b)
trans = umap.UMAP().fit(b)

cm = np.ones(b.shape[0])
cm[:b_human.shape[0]] = 0

plt.rcParams["figure.figsize"] = (2.5,2.5)
plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], c=cm, cmap=LinearSegmentedColormap.from_list('test', ['#ff7f0e', '#1f77b4'], N=2), alpha=0.8,s=20) #
custom_lines = [Line2D([0], [0], color='#ff7f0e', marker='o', linestyle=''),
                Line2D([0], [0], color='#1f77b4', marker='o', linestyle='')]
plt.legend(custom_lines, ['Humans', r'RR-RL$^2$'], bbox_to_anchor=(0.0,1.02,1,0.2), loc="lower left", borderaxespad=-0.25, ncol=2, frameon=False, handletextpad=0.1, mode='expand') #fontsize=8

plt.xlim(trans.embedding_[:, 0].min() - 0.5, trans.embedding_[:, 0].max() + 0.5)
plt.ylim(trans.embedding_[:, 1].min() - 0.5, trans.embedding_[:, 1].max() + 0.5)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

sns.despine()
plt.tight_layout()
plt.savefig('figures/gershman_embedding.pdf', bbox_inches='tight')
plt.show()
