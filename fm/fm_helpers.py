import time
from typing import Any

import torch
import wandb
from ema_pytorch import EMA
from loguru import logger
from torch_geometric.utils import to_dense_batch

from evaluation.moses.metrics import get_all_metrics
from evaluation.moses.molecules import graph_to_smiles
from evaluation.synthetic import (
    EgoSamplingMetrics,
    PlanarSamplingMetrics,
    PriceSamplingMetrics,
    ProteinSamplingMetrics,
    SBMSamplingMetrics,
    TPUSamplingMetrics,
    TreeSamplingMetrics,
)
from fm import VParamInterpolant, X0ParamInterpolant, X1ParamInterpolant
from models import AutoencoderKL, DecoderDirected, DecoderUnDirected, DiT, EncoderLPE, EncodermLPE


def get_selection_metric(cfg: Any) -> str:
    if "selection_metric" in cfg:
        return cfg.selection_metric
    if cfg.dataset.molecular:
        return "valid"
    if cfg.dataset.directed:
        return "sampling/frac_unic_non_iso_valid"
    return "acc" if cfg.dataset.name in {"planar", "tree"} else "ratio"


def get_selection_mode(cfg: Any) -> str:
    if "selection_mode" in cfg:
        return cfg.selection_mode
    return "max" if get_selection_metric(cfg) in {"valid", "acc", "sampling/frac_unic_non_iso_valid"} else "min"


def should_save_every_eval(cfg: Any) -> bool:
    return bool(cfg.get("save_every_eval", cfg.dataset.directed))


def build_sampling_metrics(
    cfg: Any,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    test: bool,
) -> Any:
    if cfg.dataset.molecular:
        return None

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if cfg.dataset.directed:
        if cfg.dataset.name == "tpu_tile":
            kwargs = {"test": test}
            if test:
                kwargs["root"] = cfg.root
            return TPUSamplingMetrics(dataloaders, **kwargs)
        if cfg.dataset.name == "price":
            return PriceSamplingMetrics(dataloaders)
    else:
        if cfg.dataset.name == "planar":
            return PlanarSamplingMetrics(dataloaders, test=test)
        if cfg.dataset.name == "sbm":
            return SBMSamplingMetrics(dataloaders, test=test)
        if cfg.dataset.name == "tree":
            return TreeSamplingMetrics(dataloaders, test=test)
        if cfg.dataset.name == "ego":
            return EgoSamplingMetrics(dataloaders, test=test)
        if cfg.dataset.name == "protein":
            return ProteinSamplingMetrics(dataloaders, test=test)

    raise NotImplementedError(f"Unknown FM dataset: {cfg.dataset.name}")


def build_autoencoder(cfg: Any, device_id: Any) -> AutoencoderKL:
    if cfg.dataset.directed:
        encoder = EncodermLPE(
            num_node_types=cfg.dataset.num_node_types,
            num_node_features=cfg.dataset.num_node_features,
            num_edge_features=cfg.dataset.num_edge_types,
            global_cfg=cfg.encoder,
            phi_cfg=cfg.encoder.phi,
            rho_cfg=cfg.encoder.rho,
            dropout=cfg.dropout,
        )
        decoder = DecoderDirected(
            pe_dim=cfg.encoder.pe_dim,
            max_num_nodes=cfg.dataset.num_nodes,
            ds_dim=cfg.encoder.ds_dim,
            num_node_features=cfg.dataset.num_node_types,
            num_edge_features=cfg.dataset.num_edge_types,
            dropout=cfg.dropout,
        )
    else:
        encoder = EncoderLPE(
            num_node_types=cfg.dataset.num_node_types,
            num_node_features=cfg.dataset.num_node_features,
            num_edge_features=cfg.dataset.num_edge_types if not cfg.dataset.molecular else cfg.dataset.num_edge_features,
            global_cfg=cfg.encoder,
            phi_cfg=cfg.encoder.phi,
            rho_cfg=cfg.encoder.rho,
            dropout=cfg.dropout,
        )
        decoder = DecoderUnDirected(
            pe_dim=cfg.encoder.pe_dim,
            max_num_nodes=cfg.dataset.num_nodes,
            ds_dim=cfg.encoder.ds_dim,
            num_node_features=cfg.dataset.num_node_types,
            num_edge_features=cfg.dataset.num_edge_types,
            dropout=cfg.dropout,
            use_rms_norm=cfg.dataset.get("decoder_use_rms_norm", False),
        )

    return AutoencoderKL(encoder, decoder, cfg.dataset.directed).to(device_id)


def build_interpolant(cfg: Any, device_id: Any) -> Any:
    interpolant_kwargs = dict(
        num_timesteps=cfg.num_sampling_steps,
        time_density=cfg.time_density,
        time_density_params=cfg.density_params,
        sampling_time_density=cfg.sampling_time_density,
        sampling_time_density_params=cfg.sampling_density_params,
        conditioning=cfg.sc_prob > 0,
        device=device_id,
    )
    if cfg.param == "x_0":
        return X0ParamInterpolant(**interpolant_kwargs)
    if cfg.param == "x_1":
        return X1ParamInterpolant(**interpolant_kwargs)
    return VParamInterpolant(**interpolant_kwargs)


def build_denoiser(cfg: Any, device_id: Any) -> DiT:
    return DiT(
        num_layers=cfg.denoiser.num_layers,
        num_heads=cfg.denoiser.num_heads,
        in_dim=cfg.encoder.pe_dim,
        embed_dim=cfg.denoiser.embed_dim,
        dropout=cfg.dropout,
        use_pe=cfg.denoiser.use_pe,
    ).to(device_id)


def build_ema_model(cfg: Any, device_id: Any) -> EMA:
    model = build_denoiser(cfg, device_id)
    return EMA(model, beta=0.9999, update_after_step=1000, update_every=1, allow_different_devices=True)


def sample_n_nodes(node_distribution: Any, cfg: Any, device_id: Any) -> torch.Tensor:
    sampled = node_distribution.sample_n(cfg.num_samples, device=device_id)
    if torch.is_tensor(sampled):
        n_nodes = sampled.to(device_id)
    else:
        n_nodes = torch.tensor(sampled, device=device_id)
    return n_nodes.long()


def generate_samples(
    interpolant: Any,
    model: Any,
    autoencoder: AutoencoderKL,
    scale_factor: float,
    node_distribution: Any,
    cfg: Any,
    device_id: Any,
    atom_decoder: Any | None = None,
    q: float | None = None,
) -> list[Any]:
    generated_samples: list[Any] = []
    sampling_start = time.time()
    for _ in range(cfg.num_sample_batch):
        n_nodes = sample_n_nodes(node_distribution, cfg, device_id)  # (bs,)
        batch_size = len(n_nodes)
        n_max = int(torch.max(n_nodes).detach().cpu())
        arange = torch.arange(n_max, device=device_id).unsqueeze(0).expand(batch_size, -1)  # (bs, n)
        sample_mask = arange < n_nodes.unsqueeze(1)  # (bs, n)
        graph_indices = torch.arange(batch_size, device=device_id).repeat_interleave(n_max).reshape(batch_size, n_max)  # (bs, n)
        batch_tensor = graph_indices[sample_mask]  # (n_total,)

        latent_samples = interpolant.sample(
            batch_size=batch_size,
            num_tokens=n_max,
            embed_dim=cfg.encoder.pe_dim,
            model=model,
            token_mask=sample_mask,
        )
        latent_tokens = latent_samples["tokens_traj"][-1] / scale_factor  # (bs, n, pe_dim)

        decode_kwargs = {"batch": batch_tensor}
        if cfg.dataset.directed:
            decode_kwargs["q"] = torch.full((batch_size,), fill_value=q, device=device_id)  # (bs,)
        decoded = autoencoder.decode(latent_tokens[sample_mask], **decode_kwargs)

        if cfg.dataset.molecular:
            e_hat, x_hat, _, _ = decoded
            x_hat, _ = to_dense_batch(x_hat, batch=batch_tensor)
            generated_samples.extend(
                graph_to_smiles(
                    e_hat=e_hat,
                    x_hat=x_hat,
                    num_nodes=n_nodes,
                    atom_decoder=atom_decoder,
                )
            )
        elif cfg.dataset.directed:
            e_hat, x_hat, edge_mask, _ = decoded
            e_hat = e_hat.argmax(dim=-1)
            e_hat = (e_hat * edge_mask).long()
            x_hat, _ = to_dense_batch(x_hat, batch=batch_tensor)
            x_hat = x_hat.argmax(dim=-1).long()
            for i in range(batch_size):
                generated_samples.append((x_hat[i, : n_nodes[i]], e_hat[i, : n_nodes[i], : n_nodes[i]]))
        else:
            e_hat, _, edge_mask, _ = decoded
            e_hat = e_hat.argmax(dim=-1)
            e_hat = (e_hat * edge_mask).long()
            for i in range(batch_size):
                generated_samples.append(e_hat[i, : n_nodes[i], : n_nodes[i]])

    sampling_end = time.time()
    total_samples = cfg.num_samples * cfg.num_sample_batch
    logger.info(
        f"Sampling {cfg.num_sample_batch} batches took : {sampling_end - sampling_start} - "
        f"Samples per second: {total_samples / (sampling_end - sampling_start)}"
    )
    return generated_samples


def evaluate_samples(
    generated_samples: list[Any],
    cfg: Any,
    device_id: Any,
    step: int,
    sampling_metrics: Any | None = None,
    train_dataset: Any | None = None,
    test_dataset: Any | None = None,
) -> dict[str, float]:
    if cfg.dataset.molecular:
        try:
            metrics = get_all_metrics(
                gen=generated_samples,
                k=cfg.num_samples * cfg.num_sample_batch,
                device=device_id,
                n_jobs=8,
                test=list(test_dataset.smiles),
                train=list(train_dataset.smiles),
            )
        except ZeroDivisionError:
            logger.info(f"No valid molecule on step {step}")
            metrics = {"valid": -1.0}
    else:
        metrics = sampling_metrics(generated_graphs=generated_samples)

    if cfg.wandb_project is not None:
        wandb.log(metrics, step=step)
    return metrics
