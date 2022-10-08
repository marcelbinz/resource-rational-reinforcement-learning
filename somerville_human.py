
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from blr import BLR

plt.rcParams["figure.figsize"] = (3.5, 3)
linspace = torch.from_numpy(np.linspace(10, 30, 1000)[:, None])

data_directed = torch.from_numpy(np.genfromtxt('data/somerville_directed.csv',delimiter=','))
data_random = torch.from_numpy(np.genfromtxt('data/somerville_random.csv',delimiter=','))

plt.scatter(data_directed[:, 0], data_directed[:, 1],s=10, alpha=0.3)
plt.scatter(data_random[:, 0], data_random[:, 1],s=10, alpha=0.3)

model = BLR(1, 100, normalize=True, polynomials=1)
model.fit(data_directed[:, [0]], data_directed[:, [1]])
mean_pred, std_pred = model.predict(linspace)
plt.plot(linspace, mean_pred, color='#1f77b4')

plt.fill_between(
    linspace[:, 0],
    mean_pred[:, 0] - std_pred[:, 0],
    mean_pred[:, 0] + std_pred[:, 0],
    alpha=0.3,
    color='#1f77b4'
)

plt.xlabel('Age')
plt.ylabel(r'$\Delta$ p(explore)')
sns.despine()
plt.xlim(12, 25)
plt.ylim(-0.1, 0.3)

model = BLR(1, 100, normalize=True, polynomials=1)
model.fit(data_random[:, [0]], data_random[:, [1]])
mean_pred, std_pred = model.predict(linspace)
plt.plot(linspace, mean_pred, color='#ff7f0e')

plt.fill_between(
    linspace[:, 0],
    mean_pred[:, 0] - std_pred[:, 0],
    mean_pred[:, 0] + std_pred[:, 0],
    alpha=0.3,
    color='#ff7f0e'
)

plt.xlabel('Age')
plt.ylabel(r'$\Delta$ p(explore)')
sns.despine()
plt.xlim(12, 28.5)
plt.ylim(-0.1, 0.3)
plt.legend(['directed', 'random'], bbox_to_anchor=(0.0,1.02,1,0.2), loc="lower left", borderaxespad=0, ncol=2, frameon=False, handletextpad=0.5, mode='expand')
plt.tight_layout()
plt.savefig('figures/somerville_human.pdf', bbox_inches='tight')
plt.show()
