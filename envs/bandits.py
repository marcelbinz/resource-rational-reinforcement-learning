import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from torch.distributions import Normal, Bernoulli, MultivariateNormal
import torch.nn.functional as F
import torch

class VectorizedNormalBandit(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, max_steps, num_actions, reward_variance, mean_variance, reward_scaling=1, batch_size=32):

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.reward_scaling = reward_scaling

        self.max_steps = max_steps
        self.reward_variance = reward_variance
        self.mean_variance = mean_variance

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(np.ones(self.num_actions), np.ones(self.num_actions))

    def reset(self):
        self.t = 0
        self.mean_reward = Normal(torch.zeros(self.batch_size, self.num_actions, device=self.device), math.sqrt(self.mean_variance) * torch.ones(self.batch_size, self.num_actions, device=self.device)).sample()
        self.rewards = Normal(self.mean_reward, math.sqrt(self.reward_variance)).sample((self.max_steps,))

        return torch.zeros(self.batch_size, self.num_actions, device=self.device)

    def step(self, action):
        # action is long
        self.t += 1
        done = True if (self.t >= self.max_steps) else False
        regrets = self.mean_reward.max(dim=1)[0] - self.mean_reward.gather(1, action.unsqueeze(1)).squeeze(1)
        reward = self.rewards[self.t - 1].gather(1, action.unsqueeze(1)).squeeze(1)
        reward = reward / self.reward_scaling

        observation = torch.stack((reward, action.float()), dim=1)

        return observation, reward, done, {'regrets': regrets.mean()}

class HorizonTaskWilson(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, num_actions, reward_scaling, reward_std, num_forced_choice, reward_shifting=0, batch_size=32):
        self.num_actions = num_actions

        self.num_forced_choice = num_forced_choice
        self.reward_scaling = reward_scaling
        self.reward_shifting = reward_shifting
        self.batch_size = batch_size
        self.reward_std = reward_std

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(np.ones(self.num_actions + 1), np.ones(self.num_actions + 1))

    def reset(self, unequal=None, horizon=None):
        self.t = 0

        if horizon is None:
            self.max_steps =  np.random.choice([1, 6])
        else:
            self.max_steps = horizon

        self.mean_reward = torch.empty(self.batch_size, self.num_actions, device=self.device)
        for i in range(self.batch_size):
            idx = np.random.choice([0, 1])
            rew = np.random.choice([40, 60])
            div = np.random.choice([-30, -20, -12, -8, -4, 4, 8, 12, 20, 30])
            self.mean_reward[i, idx] = rew
            self.mean_reward[i, 1 - idx] = rew + div

        self.rewards = Normal(self.mean_reward, self.reward_std).sample((self.num_forced_choice + self.max_steps,))

        return self.forced_choice_data(unequal)

    def forced_choice_data(self, unequal):
        # randomly select actions
        action = torch.zeros(self.num_forced_choice, self.batch_size, device=self.device).long()
        if unequal is None:
            unequal = np.random.choice([True, False], size=self.batch_size)
        for i in range(self.batch_size):
            if unequal[i]:
                # case unequal information
                options = np.array([
                    [0, 0, 0, 1],
                    [0, 1, 1, 1]])
                forced_choices = options[np.random.randint(options.shape[0])]
            else:
                # case equal information
                forced_choices = np.array([0, 0, 1, 1])

            np.random.shuffle(forced_choices)
            action[:, i] = torch.from_numpy(forced_choices).to(self.device)

        reward = torch.stack([self.rewards[t].gather(1, action[t].unsqueeze(1)).squeeze(1) for t in range(self.num_forced_choice)])
        reward = (reward - self.reward_shifting) / self.reward_scaling

        if self.max_steps == 1:
            time_step = torch.zeros(self.num_forced_choice, self.batch_size, 1).to(self.device)
        else:
            time_step = torch.ones(self.num_forced_choice, self.batch_size, 1).to(self.device)

        observation = torch.cat((
            reward.unsqueeze(-1),
            action.float().unsqueeze(-1),
            time_step
        ), dim=-1)

        return observation

    def step(self, action):
        # action is long
        self.t += 1
        done = True if (self.t >= self.max_steps) else False

        regrets = self.mean_reward.max(dim=1)[0] - self.mean_reward.gather(1, action.unsqueeze(1)).squeeze(1)
        reward = self.rewards[self.num_forced_choice + self.t - 1].gather(1, action.unsqueeze(1)).squeeze(1)
        reward = (reward - self.reward_shifting) / self.reward_scaling

        if self.max_steps == 1:
            time_step = torch.zeros(self.batch_size, 1).to(self.device)
        else:
            time_step = torch.ones(self.batch_size, 1).to(self.device)

        observation = torch.cat((
            reward.unsqueeze(-1),
            action.float().unsqueeze(-1),
            time_step
        ), dim=-1)

        return observation, reward, done, {'regrets': regrets.mean()}

class MixtureTask(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, max_steps, num_actions, reward_scaling, batch_size=32):
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.reward_scaling = reward_scaling

        self.batch_size = batch_size

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(np.ones(self.num_actions + 2), np.ones(self.num_actions + 2))

    def reset(self):
        self.t = 0

        # rewards
        self.mean_reward = torch.empty(self.batch_size, self.num_actions, device=self.device).uniform_(0, 150)

        # punishment
        self.prob_punishment = torch.empty(self.batch_size, self.num_actions, device=self.device).uniform_(0.05, 0.95)
        self.mean_punishment = torch.empty(self.batch_size, self.num_actions, device=self.device).uniform_(0, 150) / self.prob_punishment
        self.punishments = Bernoulli(self.prob_punishment).sample((self.max_steps,)) * Normal(self.mean_punishment, 10).sample((self.max_steps,))

        # average
        self.mean_total = self.mean_reward - self.prob_punishment * self.mean_punishment

        return torch.zeros(self.batch_size,self.num_actions + 2, device=self.device)

    def step(self, action):
        # action is long
        self.t += 1
        done = True if (self.t >= self.max_steps) else False

        regrets = self.mean_total.max(dim=1)[0] - self.mean_total.gather(1, action.unsqueeze(1)).squeeze(1)
        reward = self.mean_reward.gather(1, action.unsqueeze(1)).squeeze(1)
        reward = reward / self.reward_scaling
        punishment = self.punishments[self.t - 1].gather(1, action.unsqueeze(1)).squeeze(1)
        punishment = punishment / self.reward_scaling

        observation = torch.cat((
            reward.unsqueeze(-1),
            punishment.unsqueeze(-1),
            F.one_hot(action, num_classes=self.num_actions).float(),
        ), dim=-1)

        return observation, reward - punishment, done, {'regrets': regrets.mean()}

class IowaGamblingTask(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, max_steps, num_actions, reward_scaling, batch_size=32):
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.batch_size = batch_size

        self.reward_scaling = reward_scaling

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(np.ones(self.num_actions + 2), np.ones(self.num_actions + 2))

    def reset(self):
        self.t = 0
        self.punishment_a = np.array([150, 300, 200, 250, 350, 0, 0, 0, 0, 0])
        self.punishment_b = np.array([1250, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.punishment_c = np.array([50, 50, 50, 50, 50, 0, 0, 0, 0, 0])
        self.punishment_d = np.array([250, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.reward_values = torch.Tensor([100, 100, 50, 50], device=self.device)

        return torch.zeros(self.batch_size,self.num_actions + 2, device=self.device)

    def step(self, action):
        # action is long
        if not self.t % 10:
            np.random.shuffle(self.punishment_a)
            np.random.shuffle(self.punishment_b)
            np.random.shuffle(self.punishment_c)
            np.random.shuffle(self.punishment_d)
        self.t += 1
        done = True if (self.t >= self.max_steps) else False

        regrets = 0
        reward = torch.zeros(self.batch_size, device=self.device)
        punishment = torch.zeros(self.batch_size, device=self.device)

        for i in range(self.batch_size):
            reward[i] = self.reward_values[action[i]]
            if action[i] == 0:
                punishment[i] = self.punishment_a[self.t % 10]
            elif action[i] == 1:
                punishment[i] = self.punishment_b[self.t % 10]
            elif action[i] == 2:
                punishment[i] = self.punishment_c[self.t % 10]
            elif action[i] == 3:
                punishment[i] = self.punishment_d[self.t % 10]

            regrets += torch.zeros([])

        observation = torch.cat((
            reward.unsqueeze(-1) / self.reward_scaling,
            punishment.unsqueeze(-1) / self.reward_scaling,
            F.one_hot(action, num_classes=self.num_actions).float()
        ), dim=-1)

        return observation, (reward - punishment) / self.reward_scaling, done, {'regrets': regrets / self.batch_size}
