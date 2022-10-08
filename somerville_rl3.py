
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from blr import BLR

plt.rcParams["figure.figsize"] = (3.5, 3)

load_path_end = '.pth'
save_path1 = 'data/directed_rl3.csv'
save_path2 = 'data/random_rl3.csv'

klds = np.arange(1, 100, 1) * 100
klds2 = np.arange(0, 100, 1) * 100
linspace = torch.from_numpy(klds[:, None]).float()
linspace2 = torch.from_numpy(klds2[:, None]).float()

directed_exploration = torch.stack([torch.load('data/somerville_a2c' + str(float(i)) +  load_path_end)[0] for i in klds])
directed_h1 = directed_exploration.mean(1)[:, 0]
directed_h6 = directed_exploration.mean(1)[:, 1]

probs = np.concatenate([directed_h1.numpy(), directed_h6.numpy()])
horizon = np.concatenate([np.zeros(99), np.ones(99)])
age = np.concatenate([klds, klds])
directed_csv = np.stack([probs, horizon, age], axis=1)
np.savetxt(save_path1, directed_csv, header="probs,horizon,age", delimiter=",", comments='')

data_directed = (directed_h6 - directed_h1).unsqueeze(1)

model = BLR(1, 100, normalize=True, polynomials=1)
model.fit(linspace, data_directed)
mean_pred, std_pred = model.predict(linspace2)
plt.plot(linspace2, mean_pred, color='#1f77b4')

plt.fill_between(
    linspace2[:, 0],
    mean_pred[:, 0] - std_pred[:, 0],
    mean_pred[:, 0] + std_pred[:, 0],
    alpha=0.3,
    color='#1f77b4'
)

plt.scatter(klds, data_directed[:, 0],s=10, alpha=0.3)
plt.xlabel('KL Divergence')
plt.ylabel(r'$\Delta$ p(explore)')
sns.despine()
plt.xlim(100, 5000)
plt.ylim(-0.1, 0.3)

random_exploration = torch.stack([torch.load('data/somerville_a2c' + str(float(i)) +  load_path_end)[1] for i in klds])
random_h1 = random_exploration.mean(1)[:, 0]
random_h6 = random_exploration.mean(1)[:, 1]

probs = np.concatenate([random_h1.numpy(), random_h6.numpy()])
horizon = np.concatenate([np.zeros(99), np.ones(99)])
age = np.concatenate([klds, klds])
directed_csv = np.stack([probs, horizon, age], axis=1)
np.savetxt(save_path2, directed_csv, header="probs,horizon,age", delimiter=",", comments='')

data_random = (random_h6 - random_h1).unsqueeze(1)

model = BLR(1, 100, normalize=True, polynomials=1)
model.fit(linspace, data_random)
mean_pred, std_pred = model.predict(linspace2)
plt.plot(linspace2, mean_pred, color='#ff7f0e')

plt.fill_between(
    linspace2[:, 0],
    mean_pred[:, 0] - std_pred[:, 0],
    mean_pred[:, 0] + std_pred[:, 0],
    alpha=0.3,
    color='#ff7f0e'
)

plt.scatter(klds, data_random[:, 0],s=10, alpha=0.3)
plt.xlabel('Description Length')
plt.ylabel(r'$\Delta$ p(explore)')
sns.despine()
plt.xlim(0, 10000)
plt.ylim(-0.1, 0.3)
plt.legend(['directed', 'random'], bbox_to_anchor=(0.0,1.02,1,0.2), loc="lower left", borderaxespad=0, ncol=2, frameon=False, handletextpad=0.5, mode='expand')
plt.tight_layout()
plt.xticks([0, 5000, 10000])
plt.savefig('figures/somerville_rl3.pdf', bbox_inches='tight')
plt.show()
