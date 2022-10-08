import torch
from torch.distributions import Normal

class BLR():
    def __init__(self, num_features, precision=1, alpha=1, normalize=False, polynomials=1):
        self.normalize = normalize
        self.polynomials = polynomials

        if num_features == 1:
            self.num_features = polynomials + 1

        self.precision = precision
        self.alpha = alpha
        self.mean = torch.zeros(self.num_features, 1)
        self.covariance = torch.eye(self.num_features)

    def fit(self, inputs, targets):
        # inputs: (batch_size, self.num_features)
        # targets: (batch_size, 1)
        batch_size = inputs.shape[0]
        num_features = inputs.shape[1]

        if self.normalize:
            self.inputs_mean = inputs.mean(0)
            self.inputs_std = inputs.std(0)
            inputs = (inputs - self.inputs_mean) / self.inputs_std
        if num_features == 1:
            inputs = torch.pow(inputs, torch.arange(self.polynomials + 1))

        for i in range(1000):
            covariance = self.covariance / self.alpha

            posterior_covariance = torch.inverse(torch.inverse(covariance) + self.precision * (inputs.t() @ inputs))
            posterior_mean = posterior_covariance @ (torch.inverse(covariance) @ self.mean + self.precision * (inputs.t() @ targets))

            # evidence approximation
            self.alpha = num_features / (posterior_mean.t() @ posterior_mean).item()
            self.precision = (batch_size / (targets - inputs @ posterior_mean).pow(2).sum()).item()

        self.mean = posterior_mean
        self.covariance = posterior_covariance


    def predict(self, inputs):
        batch_size = inputs.shape[0]
        num_features = inputs.shape[1]

        if self.normalize:
            inputs = (inputs - self.inputs_mean) / self.inputs_std

        if num_features == 1:
            inputs = torch.pow(inputs, torch.arange(self.polynomials + 1))

        cov_pred = [(inputs[[i], :] @ self.covariance @ inputs[[i], :].t()) for i in range(batch_size)]
        return inputs @ self.mean, torch.cat(cov_pred).sqrt()
