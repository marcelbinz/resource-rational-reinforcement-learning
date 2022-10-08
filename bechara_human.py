import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import argparse

plt.rcParams["figure.figsize"] = (3,3)

means = np.zeros((4, 2))

means[0, 0] = 0.14
means[0, 1] = 0.31
means[1, 0] = 0.16
means[1, 1] = 0.31
means[2, 0] = 0.34
means[2, 1] = 0.17
means[3, 0] = 0.36
means[3, 1] = 0.21
label1 = 'Healthy'
label2 = 'Lesion'

x = np.arange(2)
width = 0.35

print(means[2, 0] + means[3, 0])
plt.bar(x - width/2, [means[2, 0] + means[3, 0], means[2, 1] + means[3, 1]], width, label='low-risk', alpha=0.8)
plt.bar(x + width/2, [means[0, 0] + means[1, 0], means[0, 1] + means[1, 1]], width, label='high-risk', alpha=0.8)
sns.despine()
plt.legend(bbox_to_anchor=(0.0,1.02,1,0.2), loc="lower left", ncol=3, frameon=False, handletextpad=0.5, mode='expand')
plt.tight_layout()
plt.ylim(0, 1.15)
plt.ylabel('Choice Probability')
plt.xticks(np.arange(2), [label1, label2])

plt.savefig('figures/bechara_human.pdf', bbox_inches='tight')
plt.show()
