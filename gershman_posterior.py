import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import math
import torch
import numpy as np
model_index = 0

plt.rcParams["figure.figsize"] = (3.75,2.)
ylabels = [r'RR-RL$^2$', r'Hybrid', 'Boltzmann', 'UCB', 'Thompson', r'RL$^2$']

logprobs = torch.from_numpy(torch.load('data/all_models_likelihoods.pth')).t()

joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[1])
marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
posterior_logprob = joint_logprob - marginal_logprob

print('Number of participants best explained by hypothesis:')
print((torch.argmax(posterior_logprob.exp().detach(), dim=-1) == model_index).sum())

print('Number of participants better explained than 0.99:')
print((posterior_logprob.exp().detach()[:, model_index] > 0.99).sum())

all_joint_logprob = joint_logprob.sum(0, keepdim=True)
all_marginal_logprob = torch.logsumexp(all_joint_logprob, dim=1, keepdim=True)
all_posterior_logprob = all_joint_logprob - all_marginal_logprob
print('Joint posterior for each model: ' + str(all_posterior_logprob.exp()))

print(posterior_logprob.shape)

part = 0
if part == 0:
    start = 0
    end = 23
else:
    start = 22
    end = 45
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
fig.add_axes(cbar_ax)

cmap = mpl.colors.LinearSegmentedColormap.from_list('testCmap', ['#ffffff', '#363737'])
ax = sns.heatmap(posterior_logprob.exp().t().detach()[:, start:end], cmap=cmap, vmax=1.0, center=0.5, square=True, linewidths=.5, ax=ax, cbar_ax=cbar_ax, linecolor='#d8dcd6')
fig.axes[-1].yaxis.set_ticks([0, 1.0])

ax.set_yticklabels(ylabels, rotation='horizontal', size=5)
if part != 0:
    ax.set_xlabel('Participant')
    labels = [int(item.get_text()) + start for item in ax.get_xticklabels()]
    print(labels)
    ax.set_xticklabels(labels)

[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 5 != 0]
plt.tight_layout()
plt.savefig('figures/posteriors' + str(part) + '.pdf', bbox_inches='tight')
plt.show()
