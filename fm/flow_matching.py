import torch
import math
from abc import abstractmethod
from loguru import logger


class BaseFlowMatchingInterpolant:
    def __init__(
            self,
            min_t: float = 0.,
            max_t: float = 1.,
            num_timesteps: int = 100,
            time_density: str = 'linear',
            time_density_params: list = None,
            sampling_time_density: str = 'linear',
            sampling_time_density_params: list = None,
            conditioning: bool = False,
            device: str = "cpu",

    ):
        self.min_t = min_t
        self.max_t = max_t
        self.num_timesteps = num_timesteps
        self.time_density = time_density
        self.time_density_params = time_density_params
        self.sampling_time_density = sampling_time_density
        self.sampling_time_density_params = sampling_time_density_params
        self.conditioning = conditioning
        self.device = device

    def _sample_t(self, batch_size):
        if self.time_density in ['linear', 'cos', 'polydec']:
            t = torch.rand(batch_size, device=self.device)
            if self.time_density == 'linear':
                pi_t = t
            elif self.time_density == 'cos':
                pi_t = 1 - 1 / (torch.tan(math.pi / 2 * t) + 1)
            else:
                pi_t = 2 * t - t ** 2
        else:
            u = self.time_density_params[1] * torch.randn(batch_size, device=self.device) + self.time_density_params[0]
            pi_t = 1 / (1 + torch.exp(-u))
            pi_t = 0.02 * torch.rand(batch_size, device=self.device) + 0.98 * pi_t
        return pi_t * (self.max_t - 2*self.min_t) + self.min_t

    def sample_sampling_t(self):
        if self.sampling_time_density in ['linear', 'cos', 'polydec']:
            ts = torch.linspace(self.min_t, self.max_t, self.num_timesteps + 1)
            if self.sampling_time_density == 'linear':
                ts = ts
            elif self.sampling_time_density == 'cos':
                ts = 1 - 1 / (torch.tan(math.pi / 2 * ts) + 1)
            elif self.sampling_time_density == 'polydec':
                ts = 2*ts - ts**2
            else:
                raise NotImplementedError(f"Unsupported time density {self.sampling_time_density}")
            
        elif self.sampling_time_density == 'log_norm':
            u = self.sampling_time_density_params[1] * torch.randn(self.num_timesteps - 1, device=self.device) + self.sampling_time_density_params[0]
            pi_t = 1 / (1 + torch.exp(-u))
            pi_t = 0.02 * torch.rand(self.num_timesteps - 1, device=self.device) + 0.98 * pi_t
            pi_t = pi_t * (self.max_t - 2 * self.min_t) + self.min_t
            ts = torch.sort(torch.clamp(pi_t, 2 * self.min_t, self.max_t))[0]
            ts = torch.cat([
                torch.tensor([self.min_t], device=self.device),
                ts,
                torch.tensor([self.max_t], device=self.device)
            ])
        else:
            raise NotImplementedError(f"Unsupported time density {self.sampling_time_density}.")
        return ts

    def gaussian(self, batch_size, num_tokens, emb_dim=3):
        noise = torch.randn(batch_size, num_tokens, emb_dim, device=self.device)
        return noise

    def _corrupt_x(self, x_1, t, token_mask, diffuse_mask):
        x_0 = self.gaussian(*x_1.shape)
        x_t = (1 - t[..., None]) * x_0 + t[..., None] * x_1
        x_t = x_t * diffuse_mask[..., None] + x_1 * (~diffuse_mask[..., None])
        return x_t * token_mask[..., None], x_0

    def corrupt_batch(self, x_1, mask):
        """
            Corrupts a batch of data by sampling a time t and interpolating to noisy samples.

        :param x_1: (bs, n, d)
        :param mask: (bs, n)
        :return:
        """
        # [B, N]
        token_mask = mask
        diffuse_mask = mask
        batch_size, _ = diffuse_mask.shape

        # [B, 1]
        t = self._sample_t(batch_size)[:, None]
        x_t, x_0 = self._corrupt_x(x_1, t, token_mask, diffuse_mask)

        if torch.any(torch.isnan(x_t)):
            raise ValueError("NaN in x_t during corruption")

        return x_t, t, x_0

    @abstractmethod
    def _x_vector_field(self, t: torch.Tensor, prediction: torch.Tensor, x_t: torch.Tensor):
        pass

    def _x_euler_step(self, d_t: torch.Tensor, t: torch.Tensor, prediction: torch.Tensor, x_t: torch.Tensor):
        assert d_t > 0
        x_vf = self._x_vector_field(t, prediction, x_t)
        return x_t + x_vf * d_t

    def criterion(self, gt, t, pred_x, mask):
        pass

    def sample(
            self,
            batch_size,
            num_tokens,
            embed_dim,
            model,
            num_timesteps=None,
            token_mask=None,
    ):
        """Generates new samples of a specified (B, N, d) using denoiser model.

        Args:
            batch_size (int): Number of samples to generate.
            num_tokens (int): Number of tokens in each sample.
            emb_dim (int): Dimension of each token.
            model (nn.Module): Denoiser model to use.
            num_timesteps (int): Number of timesteps to integrate over.
            token_mask (torch.Tensor): Mask for valid tokens.

        Returns:
            Dict with keys:
                tokens_traj (list): List of generated samples at each timestep.
                clean_traj (list): List of denoised samples at each timestep.
        """
        x_0 = self.gaussian(batch_size, num_tokens, embed_dim)
        ts = self.sample_sampling_t()
        t_1 = ts[0]

        tokens_traj = [x_0]
        clean_traj = [None]
        for t_2 in ts[1:]:
            # Run denoiser model
            x_t_1 = tokens_traj[-1]
            t = torch.ones((batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1

            with torch.no_grad():
                # Epsilon hat
                pred_x_1 = model(x_t_1, t, token_mask, clean_traj[-1] if self.conditioning else None)

            # Process model output
            clean_traj.append(pred_x_1)

            # Take reverse step
            try:
                x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)
            except:
                print(t_2, t_1)
                assert False
            tokens_traj.append(x_t_2)
            t_1 = t_2

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}


class X0ParamInterpolant(BaseFlowMatchingInterpolant):
    def __init__(
            self,
            num_timesteps: int = 100,
            time_density: str = 'linear',
            time_density_params: list = None,
            sampling_time_density: str = 'linear',
            sampling_time_density_params: list = None,
            conditioning: bool = False,
            device: str = "cpu",
    ):
        super().__init__(
            min_t=1e-2,
            max_t=1.,
            num_timesteps=num_timesteps,
            time_density=time_density,
            time_density_params=time_density_params,
            sampling_time_density=sampling_time_density,
            sampling_time_density_params=sampling_time_density_params,
            conditioning=conditioning,
            device=device)

    def _x_vector_field(self, t, x_0, x_t):
        return (x_t - x_0) / t

    def criterion(self, x_0, t, pred_x, mask):
        # Compute MSE loss w/ masking for padded tokens
        norm_scale = torch.max(t.unsqueeze(-1), torch.tensor(0.1))
        x_error = (pred_x - x_0) / norm_scale
        loss_denom = torch.sum(mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error ** 2 * mask[..., None], dim=(-1, -2)) / loss_denom
        return x_loss.mean()

    def sample_with_energy_guidance(self,
            batch_size,
            num_tokens,
            embed_dim,
            model,
            decoder,
            scale_factor,
            q,
            batch_tensor,
            num_timesteps=None,
            token_mask=None):
        """Generates new samples of a specified (B, N, d) using denoiser model.

        Args:
            batch_size (int): Number of samples to generate.
            num_tokens (int): Number of tokens in each sample.
            emb_dim (int): Dimension of each token.
            model (nn.Module): Denoiser model to use.
            num_timesteps (int): Number of timesteps to integrate over.
            token_mask (torch.Tensor): Mask for valid tokens.

        Returns:
            Dict with keys:
                tokens_traj (list): List of generated samples at each timestep.
                clean_traj (list): List of denoised samples at each timestep.
        """
        x_0 = self.gaussian(batch_size, num_tokens, embed_dim)
        ts = self.sample_sampling_t()
        t_1 = ts[0]

        tokens_traj = [x_0]
        clean_traj = [None]
        for t_2 in ts[1:]:
            # Run denoiser model
            x_t_1 = tokens_traj[-1]
            t = torch.ones((batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1

            with torch.no_grad():
                pred_x_1 = model(x_t_1, t, token_mask, clean_traj[-1] if self.conditioning else None)

            # Process model output
            clean_traj.append(pred_x_1)

            # Compute energy
            x_t_req = x_t_1.detach().requires_grad_(True)
            x_1_hat = (x_t_req - (1 - t[..., None]) * pred_x_1.detach()) / t[..., None]
            x_1_hat /= scale_factor
            e_hat, x_hat, edge_mask, node_mask = decoder(x_1_hat[token_mask], batch=batch_tensor,
                                                 q=torch.full((batch_size, ), fill_value=q))

            delta = e_hat[..., 1] - e_hat[..., 0]  # edge - no_edge
            W = torch.sigmoid(delta / 2)

            W = W * edge_mask.to(W.dtype)  # (bs, n, n)
            W2 = W * W  # (bs, n, n)
            lam_up = W2.sum(-1).max(-1).values.clamp_min(1.0)  # (B,)
            tau = (20. / lam_up).clamp(max=1.0).view(-1, 1, 1)  # (B,1,1)

            n_nodes = node_mask.sum(dim=-1)
            exp_W2 = torch.matrix_exp(tau * W2)  # (bs, n, n)
            trace = torch.diagonal(exp_W2, dim1=-2, dim2=-1)  # (bs, n)
            h = (trace[:, :node_mask.shape[1]] * node_mask).sum(dim=-1) - n_nodes  # (bs,)

            h = (h**2).mean()
            g = torch.autograd.grad(h, x_t_req)[0]
            pred_x_1_e = pred_x_1 - t[..., None] * g

            # Take reverse step
            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1_e, x_t_1)

            tokens_traj.append(x_t_2)
            t_1 = t_2

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}



class X1ParamInterpolant(BaseFlowMatchingInterpolant):
    def __init__(
            self,
            num_timesteps: int = 100,
            time_density: str = 'linear',
            time_density_params: list = None,
            sampling_time_density: str = 'linear',
            sampling_time_density_params: list = None,
            conditioning: bool = False,
            device: str = "cpu",
    ):
        super().__init__(
            min_t=1e-2,
            max_t=1.,
            num_timesteps=num_timesteps,
            time_density=time_density,
            time_density_params=time_density_params,
            sampling_time_density=sampling_time_density,
            sampling_time_density_params=sampling_time_density_params,
            conditioning=conditioning,
            device=device)

    def _x_vector_field(self, t, x_1, x_t):
        return (x_1 - x_t) / (1 - t)

    def criterion(self, x_1, t, pred_x, mask):
        # Compute MSE loss w/ masking for padded tokens
        norm_scale = 1 - torch.min(t.unsqueeze(-1), torch.tensor(0.9))
        x_error = (x_1 - pred_x) / norm_scale
        loss_denom = torch.sum(mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error ** 2 * mask[..., None], dim=(-1, -2)) / loss_denom
        return x_loss.mean()


class VParamInterpolant(BaseFlowMatchingInterpolant):
    def __init__(
            self,
            num_timesteps: int = 100,
            time_density: str = 'linear',
            time_density_params: list = None,
            sampling_time_density: str = 'linear',
            sampling_time_density_params: list = None,
            conditioning: bool = False,
            device: str = "cpu",
    ):
        super().__init__(
            min_t=1e-2,
            max_t=1.,
            num_timesteps=num_timesteps,
            time_density=time_density,
            time_density_params=time_density_params,
            sampling_time_density=sampling_time_density,
            sampling_time_density_params=sampling_time_density_params,
            conditioning=conditioning,
            device=device)

    def _x_vector_field(self, t, v, x_t):
        return v

    def criterion(self, v, t, pred_v, mask):
        # Compute MSE loss w/ masking for padded tokens
        norm_scale = 1
        v_error = (v - pred_v) / norm_scale
        loss_denom = torch.sum(mask, dim=-1) * pred_v.size(-1)
        v_loss = torch.sum(v_error ** 2 * mask[..., None], dim=(-1, -2)) / loss_denom
        return v_loss.mean()
