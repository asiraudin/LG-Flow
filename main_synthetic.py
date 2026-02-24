import hydra
import time
import torch
from loguru import logger
import wandb
import random

from utils import (
    setup_everything,
    save_checkpoint,
    load_checkpoint,
    create_dataloaders,
    setup_training,
    instantiate_dataset
)

from models import AutoencoderKL, EncoderLPE, DecoderUnDirected
from models import DiT
from fm import X0ParamInterpolant, X1ParamInterpolant, VParamInterpolant, compute_n_nodes_distr


from torch.distributed import destroy_process_group
from torch_geometric.utils import to_dense_batch
from evaluation.synthetic import PlanarSamplingMetrics, SBMSamplingMetrics, TreeSamplingMetrics, EgoSamplingMetrics, ProteinSamplingMetrics


def evaluate(interpolant, model, autoencoder, scale_factor, node_distribution, sampling_metrics, cfg, step, device_id):
    sample_adjs = []
    sampling_start = time.time()
    for _ in range(cfg.num_sample_batch):
        n_nodes = node_distribution.sample_n(cfg.batch_size, device=device_id)  # (bs)
        n_nodes = torch.Tensor(n_nodes).long()

        # Create node mask
        batch_size = len(n_nodes)
        n_max = int(torch.max(n_nodes).detach().cpu())
        arange = torch.arange(n_max, device=device_id).unsqueeze(0).expand(batch_size, -1)
        sample_mask = arange < n_nodes.unsqueeze(1)                                             # (bs, n)

        # Create batch vector
        graph_indices = torch.arange(batch_size, device=device_id).repeat_interleave(n_max).reshape(batch_size, n_max)
        batch_tensor = graph_indices[sample_mask]

        latent_samples = interpolant.sample(
            batch_size=batch_size,
            num_tokens=n_max,
            embed_dim=cfg.encoder.pe_dim,
            model=model,
            token_mask=sample_mask,
        )
        latent_samples = latent_samples["tokens_traj"][-1]

        latent_samples /= scale_factor
        samples = autoencoder.decode(latent_samples[sample_mask], batch=batch_tensor)

        e_hat, x_hat, edge_mask, _ = samples
        e_hat = e_hat.argmax(dim=-1)
        e_hat = (e_hat * edge_mask).long()
        for i in range(batch_size):
            sample_adjs.append(e_hat[i, :n_nodes[i], :n_nodes[i]])

    sampling_end = time.time()
    logger.info(f'Sampling {cfg.num_sample_batch} batches took : {sampling_end - sampling_start} - '
                f'Samples per second: {cfg.batch_size * cfg.num_sample_batch / (sampling_end - sampling_start)}')

    metrics = sampling_metrics(
        generated_graphs=sample_adjs
    )

    if cfg.wandb_project is not None:
        wandb.log(
            metrics,
            step=step,
        )
    return metrics


@hydra.main(version_base=None, config_path="./configs", config_name="planar_fm_train")
def main(cfg):
    device, device_id, device_count, master_process, data_dir, ckpt_dir, dtype, tdtype = setup_everything(cfg)

    logger.info(f"Loading datasets from {data_dir}")
    train_dataset, val_dataset, test_dataset = instantiate_dataset(
        name=cfg.dataset.name,
        data_dir=data_dir,
        cfg=cfg,
        master_process=master_process
    )
    node_distribution = compute_n_nodes_distr(
        train_n_nodes=train_dataset.num_nodes,
        val_n_nodes=val_dataset.num_nodes,
        test_n_nodes=test_dataset.num_nodes
    )
    logger.info("Dataset loaded")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        cfg=cfg,
        device_count=device_count,
        master_process=master_process
    )
    if master_process:
        if cfg.dataset.name == 'planar':
            sampling_metrics = PlanarSamplingMetrics({'train': train_loader, 'val': val_loader, 'test': test_loader})
        elif cfg.dataset.name == 'sbm':
            sampling_metrics = SBMSamplingMetrics({'train': train_loader, 'val': val_loader, 'test': test_loader})
        elif cfg.dataset.name == 'tree':
            sampling_metrics = TreeSamplingMetrics({'train': train_loader, 'val': val_loader, 'test': test_loader})
        elif cfg.dataset.name == 'ego':
            sampling_metrics = EgoSamplingMetrics({'train': train_loader, 'val': val_loader, 'test': test_loader})
        elif cfg.dataset.name == 'protein':
            sampling_metrics = ProteinSamplingMetrics({'train': train_loader, 'val': val_loader, 'test': test_loader})
        else:
            raise NotImplementedError("Unknown dataset: {}".format(cfg.dataset.name))

    # Setup autoencoder
    encoder = EncoderLPE(
        num_node_types=cfg.dataset.num_node_types,
        num_node_features=cfg.dataset.num_node_features,
        num_edge_features=cfg.dataset.num_edge_types if not cfg.dataset.molecular else cfg.dataset.num_edge_features,
        global_cfg=cfg.encoder,
        phi_cfg=cfg.encoder.phi,
        rho_cfg=cfg.encoder.rho,
        dropout=cfg.dropout
    )

    decoder = DecoderUnDirected(
        pe_dim=cfg.encoder.pe_dim,
        max_num_nodes=cfg.dataset.num_nodes,
        ds_dim=cfg.encoder.ds_dim,
        num_node_features=cfg.dataset.num_node_types,
        num_edge_features=cfg.dataset.num_edge_types,
        dropout=cfg.dropout
    )

    autoencoder = AutoencoderKL(encoder, decoder, False).to(device_id)
    autoencoder.eval()
    load_checkpoint(
            f"{ckpt_dir}/{cfg.ae_checkpoint_file}.pt", autoencoder, device_id
    )
    for param in autoencoder.parameters():
        param.requires_grad = False

    if cfg.param == 'x_0':
        # Setup flow matching
        interpolant = X0ParamInterpolant(
            num_timesteps=cfg.num_sampling_steps,
            time_density=cfg.time_density,
            time_density_params=cfg.density_params,
            sampling_time_density=cfg.sampling_time_density,
            sampling_time_density_params=cfg.sampling_density_params,
            conditioning=cfg.sc_prob > 0,
            device=device_id
        )
    elif cfg.param == 'x_1':
        interpolant = X1ParamInterpolant(
            num_timesteps=cfg.num_sampling_steps,
            time_density=cfg.time_density,
            time_density_params=cfg.density_params,
            sampling_time_density=cfg.sampling_time_density,
            sampling_time_density_params=cfg.sampling_density_params,
            conditioning=cfg.sc_prob > 0,
            device=device_id
        )
    else:
        interpolant = VParamInterpolant(
            num_timesteps=cfg.num_sampling_steps,
            time_density=cfg.time_density,
            time_density_params=cfg.density_params,
            sampling_time_density=cfg.sampling_time_density,
            sampling_time_density_params=cfg.sampling_density_params,
            conditioning=cfg.sc_prob > 0,
            device=device_id
        )

    model = DiT(num_layers=cfg.denoiser.num_layers, num_heads=cfg.denoiser.num_heads, in_dim=cfg.encoder.pe_dim,
                embed_dim=cfg.denoiser.embed_dim, use_pe=cfg.denoiser.use_pe).to(device_id)

    model, ema_model, optimizer, lr_scheduler, step, checkpoint_file, best_val = setup_training(
        cfg=cfg,
        model=model,
        master_process=master_process,
        device=device,
        device_id=device_id,
        device_count=device_count,
        dtype=dtype,
        ckpt_dir=ckpt_dir
    )
    best_val = best_val[0]
    logger.info(
        f"Starting/resuming training for {int(cfg.num_steps) - (step - 1)} steps 🚀"
    )

    if master_process:
        start_time = time.time()
    if device_count > 1:
        train_loader.sampler.set_epoch(epoch := 0)
    first_batch = True
    scale_factor = 1.
    while step <= cfg.num_steps:
        model.train()
        loss_window = []
        for batch in train_loader:
            batch = batch.to(device_id)
            if first_batch:
                z = autoencoder.encode(batch).sample().detach() if cfg.sample else autoencoder.encode(batch).mode().detach()
                scale_factor = 1. / z.flatten().std()
                first_batch = False

            with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
                x_1 = scale_factor * autoencoder.encode(batch).sample().detach() if cfg.sample else autoencoder.encode(batch).mode().detach()
                x_1, mask = to_dense_batch(x_1, batch.batch)                                # (bs, n, pe_dim)

                x_t, t, x_0 = interpolant.corrupt_batch(
                    x_1,
                    mask,
                )

                if random.random() < cfg.sc_prob:
                    with torch.no_grad():
                        x_sc = model(x_t, t, mask)
                else:
                    x_sc = None

                out = model(x_t, t, mask, x_sc)       # (bs, n, pe_dim)

                # flow matching l2 loss
                if cfg.param == 'x_0':
                    loss = interpolant.criterion(x_0, t, out, mask)
                elif cfg.param == 'x_1':
                    loss = interpolant.criterion(x_1, t, out, mask)
                else:
                    loss = interpolant.criterion(x_1 - x_0, t, out, mask)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            if master_process:
                ema_model.update()
                loss_window.append(float(loss.detach().cpu()))

            if step % int(cfg.log_after) == 0 and master_process:
                if cfg.wandb_project is not None and master_process:
                    wandb.log(
                        dict(
                            train_loss=sum(loss_window) / len(loss_window),
                            lr=optimizer.param_groups[0]["lr"],
                            time=time.time() - start_time,
                        ),
                        step=step,
                    )
                loss_window = []
                start_time = time.time()

            if step % int(cfg.val_after) == 0 and step >= int(cfg.val_after):
                logger.info(f"Completed epoch on [{device_id}]")
                if device_count > 1:
                    epoch += 1
                    train_loader.sampler.set_epoch(epoch)
                if master_process:
                    logger.info("Evaluating model")
                    ema_model.eval()

                    metrics = evaluate(
                        interpolant,
                        ema_model,
                        autoencoder,
                        scale_factor,
                        node_distribution,
                        sampling_metrics,
                        cfg,
                        step,
                        device_id
                    )
                    target_metric = 'acc' if 'acc' in metrics else 'ratio'
                    if best_val is None or (metrics[target_metric] > best_val if target_metric == 'acc' else metrics[target_metric] < best_val):
                        best_val = metrics[target_metric]
                        if cfg.checkpoint is not None:
                            # module = model.module if device_count > 1 else model
                            module = ema_model
                            save_checkpoint(
                                checkpoint_file,
                                step,
                                module,
                                optimizer,
                                ema=True,
                                best_val=best_val
                            )

                    ema_model.train()

            step += 1

    if master_process:
        wandb.finish()

    if device_count > 1:
        destroy_process_group()


if __name__ == "__main__":
    main()
