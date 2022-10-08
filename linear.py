import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import  Normal
from torch.distributions.kl import kl_divergence
from torch.nn import init, Parameter

# https://github.com/senya-ashukha/sparse-vd-pytorch/blob/master/svdo-solution.ipynb
class LinearSVDO(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(LinearSVDO, self).__init__()
        if bias:
            self.in_features = in_features + 1
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.bias = bias

        self.weight_mu = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)
        self.log_sigma = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.normal_(0, stdv)
        self.log_sigma.data.fill_(-5)

    def forward(self, x, zeta):
        if self.bias:
            b = torch.ones(x.shape[0], 1, device=zeta.device)
            return F.linear(torch.cat((x, b), dim=1), zeta)
        else:
            return F.linear(x, zeta)

    def get_zeta(self, batch_size):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.weight_mu))
        #self.log_alpha = torch.clamp(self.log_alpha, -10, 10) # TODO

        if self.training:
            return Normal(self.weight_mu, torch.exp(self.log_sigma) + 1e-8).rsample()
        else:
            return self.weight_mu * (self.log_alpha < 3).float()

    def kl_divergence(self):
        # Return KL here -- a scalar
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha)) - k1
        a = - torch.sum(kl)
        return a
