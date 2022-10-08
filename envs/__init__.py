from gym.envs.registration import register

register(
    id='gershman2018deconstructing-v0',
    entry_point='envs.bandits:VectorizedNormalBandit',
    kwargs={'max_steps': 10, 'num_actions': 2, 'reward_variance': 1, 'mean_variance': 100, 'reward_scaling': 1.0},
)

register(
    id='somerville2017charting-v2',
    entry_point='envs.bandits:HorizonTaskWilson',
    kwargs={'num_actions': 2, 'reward_scaling': 10, 'reward_std': 8, 'num_forced_choice': 4},
)

register(
    id='bechara1994training-v0',
    entry_point='envs.bandits:MixtureTask',
    kwargs={'max_steps': 100, 'num_actions': 4, 'reward_scaling': 15},
)

register(
    id='bechara1994insensitivity-v0',
    entry_point='envs.bandits:IowaGamblingTask',
    kwargs={'max_steps': 100, 'num_actions': 4, 'reward_scaling': 15},
)
