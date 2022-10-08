import torch
import torch.nn as nn
from torch.distributions import  Normal, Categorical
from linear import LinearSVDO
import torch.nn.functional as F
from recurrent import ProbabilisticGRUCell
import numpy as np

class RL3A2C(nn.Module):
    def __init__(self, num_states, num_actions, num_hidden, prior, bias=False):
        super(RL3A2C, self).__init__()
        self.num_actions = num_actions
        self.num_hidden = num_hidden

        self.initial = nn.Parameter(0.00 * torch.randn(1, self.num_hidden), requires_grad=False)
        self.beta = nn.Parameter(-16 * torch.ones([]), requires_grad=True)

        self.gru = ProbabilisticGRUCell(num_states, num_hidden, prior, bias)

        self.mu_actor = LinearSVDO(num_hidden, num_actions, bias)
        self.mu_critic = LinearSVDO(num_hidden, 1, bias)

        self.prior = prior

    def forward(self, input, hx, zeta):
        hx = self.gru(input, hx, zeta[0])
        return self.mu_actor(hx, zeta[1]), self.mu_critic(hx, zeta[2]), hx

    def act(self, input, hx, zeta):
        logits, values, hx = self(input, hx, zeta)
        policy = Categorical(torch.nn.functional.softmax(logits, dim=-1))
        action = policy.sample()

        return policy, values, hx, action

    def initial_states(self, batch_size):
        return self.initial.expand(batch_size, -1)

    def get_zeta(self, batch_size):
        return (self.gru.get_zeta(batch_size), self.mu_actor.get_zeta(batch_size), self.mu_critic.get_zeta(batch_size))

    def kl_divergence(self):
        return self.gru.kl_divergence() + self.mu_actor.kl_divergence() + self.mu_critic.kl_divergence()
