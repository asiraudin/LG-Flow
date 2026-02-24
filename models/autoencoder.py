import torch
import torch.nn as nn


class AutoencoderKL(nn.Module):
    """
    ## Autoencoder

    This consists of the encoder and decoder modules.
    """

    def __init__(self, encoder, decoder, directed):
        """
        :param encoder: is the encoder
        :param decoder: is the decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.directed = directed

    def forward(self, data, sample_posterior=True):
        posterior = self.encode(data)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = self.decode(z, data.batch, data.q if self.directed else None)
        return *z, posterior

    def encode(self, data):
        z = self.encoder(data)
        posterior = DiagonalGaussianDistribution(z)
        return posterior

    def decode(self, z: torch.Tensor, batch, q=None):
        return self.decoder(z, batch, q if self.directed else None)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                   + self.var - 1.0 - self.logvar,
                                   dim=1)

    def mode(self):
        return self.mean


class Autoencoder(nn.Module):
    """
    ## Autoencoder

    This consists of the encoder and decoder modules.
    """

    def __init__(self, encoder, decoder):
        """
        :param encoder: is the encoder
        :param decoder: is the decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        latent = self.encode(data)
        z = self.decode(latent, data.batch)
        return *z, latent

    def encode(self, data):
        z = self.encoder(data)
        return z

    def decode(self, z: torch.Tensor, batch):
        return self.decoder(z, batch)
