import hydra
import time
import torch
import torch.nn.functional as F
from loguru import logger
import wandb

from utils import (
    setup_everything,
    save_checkpoint,
    create_dataloaders,
    setup_training,
    instantiate_dataset
)
from models import AutoencoderKL, EncoderLPE, DecoderUnDirected, EncodermLPE, DecoderDirected
from fm import compute_n_nodes_distr
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.transforms import Compose

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall
)

from torch.distributed import destroy_process_group


def _disable_sync(m):
    if m is not None:
        # TorchMetrics flags
        if hasattr(m, "sync_on_compute"):   m.sync_on_compute = False
        if hasattr(m, "dist_sync_on_step"): m.dist_sync_on_step = False


@torch.no_grad()
def evaluate(model, loader, device, num_node_types, num_edge_types, max_num_nodes):
    edge_metrics = {
        'accuracy': MulticlassAccuracy(num_classes=num_edge_types + 1, ignore_index=-1).to(device),
        'precision': MulticlassPrecision(num_classes=num_edge_types + 1, ignore_index=-1).to(device),
        'recall': MulticlassRecall(num_classes=num_edge_types + 1, ignore_index=-1).to(device),
        'f1': MulticlassF1Score(num_classes=num_edge_types + 1, ignore_index=-1).to(device),
        'sample_accuracy': MulticlassAccuracy(num_classes=num_edge_types + 1, ignore_index=-1,
                                              multidim_average='samplewise').to(device)
    }
    if num_node_types > 1:
        node_metrics = {
            'accuracy': MulticlassAccuracy(num_classes=num_node_types, ignore_index=-1).to(device),
            'precision': MulticlassPrecision(num_classes=num_node_types, ignore_index=-1).to(device),
            'recall': MulticlassRecall(num_classes=num_node_types, ignore_index=-1).to(device),
            'f1': MulticlassF1Score(num_classes=num_node_types, ignore_index=-1).to(device),
            'sample_accuracy': MulticlassAccuracy(num_classes=num_node_types, ignore_index=-1,
                                                  multidim_average='samplewise').to(device)
        }
    else:
        node_metrics = {key: None for key in edge_metrics.keys()}

    for m in edge_metrics.values(): _disable_sync(m)
    for m in node_metrics.values(): _disable_sync(m)
    with torch.inference_mode():
        for k, batch in enumerate(loader):
            batch = batch.to(device)
            edge_attr_targets = batch.edge_attr[..., 0] + 1
            node_attr_targets = batch.x[..., 0]
            with torch.no_grad():
                e_hat, x_hat, edge_mask, _, _ = model(batch, sample_posterior=False)

            e_hat, x_hat = e_hat.softmax(dim=-1), x_hat.softmax(dim=-1)  # (bs, n, n, e), (N_sum, x)
            e_hat, x_hat = e_hat.argmax(dim=-1), x_hat.argmax(dim=-1)  # (bs, n, n), (N_sum)

            e = to_dense_adj(
                edge_index=batch.edge_index,
                batch=batch.batch,
                edge_attr=edge_attr_targets
            )
            e = e.masked_fill(edge_mask == 0, -1)

            x, _ = to_dense_batch(node_attr_targets, batch.batch, fill_value=-1, max_num_nodes=max_num_nodes)
            x_hat, _ = to_dense_batch(x_hat, batch.batch, max_num_nodes=max_num_nodes)

            for (name, edge_metric), (_, node_metric) in zip(edge_metrics.items(), node_metrics.items()):
                edge_metric(e_hat, e)
                if num_node_types > 1:
                    node_metric(x_hat, x)
    
    edge_metrics = {key: value.compute() for key, value in edge_metrics.items()}
    if num_node_types > 1:
        node_metrics = {key: value.compute() for key, value in node_metrics.items()}
    num_samples = edge_metrics['sample_accuracy'].shape[0]

    edge_sample_accuracy = torch.isclose(
        edge_metrics['sample_accuracy'],
        torch.ones_like(edge_metrics['sample_accuracy']),
        rtol=1e-6
    )
    edge_metrics['sample_accuracy'] = edge_sample_accuracy.sum() / num_samples

    if num_node_types > 1:
        node_sample_accuracy = torch.isclose(
            node_metrics['sample_accuracy'],
            torch.ones_like(node_metrics['sample_accuracy']),
            rtol=1e-6
        )
        node_metrics['sample_accuracy'] = node_sample_accuracy.sum() / num_samples

        sample_accuracy = (node_sample_accuracy & edge_sample_accuracy).sum() / num_samples
    else:
        sample_accuracy = edge_metrics['sample_accuracy']

    return sample_accuracy, edge_metrics, node_metrics


@hydra.main(version_base=None, config_path="./configs", config_name="debug")
def main(cfg):
    device, device_id, device_count, master_process, data_dir, ckpt_dir, dtype, tdtype = setup_everything(cfg)

    logger.info(f"Loading datasets from {data_dir}")
    train_dataset, val_dataset, test_dataset = instantiate_dataset(
        name=cfg.dataset.name,
        data_dir=data_dir,
        cfg=cfg,
        master_process=master_process
    )

    logger.info("Dataset loaded")
    _ = compute_n_nodes_distr(
        train_n_nodes=train_dataset.num_nodes,
        val_n_nodes=val_dataset.num_nodes,
        test_n_nodes=test_dataset.num_nodes
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        cfg=cfg,
        device_count=device_count,
        master_process=master_process
    )

    if cfg.dataset.directed:
        encoder = EncodermLPE(
            num_node_types=cfg.dataset.num_node_types,
            num_node_features=cfg.dataset.num_node_features,
            num_edge_features=cfg.dataset.num_edge_types,
            global_cfg=cfg.encoder,
            phi_cfg=cfg.encoder.phi,
            rho_cfg=cfg.encoder.rho,
            dropout=cfg.dropout
            )

        decoder = DecoderDirected(
            pe_dim=cfg.encoder.pe_dim,
            max_num_nodes=cfg.dataset.num_nodes,
            ds_dim=cfg.encoder.ds_dim,
            num_node_features=cfg.dataset.num_node_types,
            num_edge_features=cfg.dataset.num_edge_types,
            dropout=cfg.dropout
        )
    else:
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

    model = AutoencoderKL(encoder, decoder, cfg.dataset.directed).to(device_id)

    model, _, optimizer, scheduler, step, checkpoint_file, best = setup_training(
        cfg=cfg,
        model=model,
        master_process=master_process,
        device=device,
        device_id=device_id,
        device_count=device_count,
        dtype=dtype,
        ckpt_dir=ckpt_dir,
        autoencoder=True
    )
    best_sample_acc, best_edge_metrics, best_node_metrics = best
    logger.info(
        f"Starting/resuming training for {int(cfg.num_steps) - (step - 1)} steps 🚀"
    )

    if master_process:
        start_time = time.time()
    if device_count > 1:
        train_loader.sampler.set_epoch(epoch := 0)
    while step <= cfg.num_steps:
        model.train()
        loss_window = []
        for batch in train_loader:
            batch = batch.to(device_id)
            with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
                edge_attr_targets = batch.edge_attr[..., 0] + 1
                node_attr_targets = batch.x.clone()
                edge_targets = to_dense_adj(
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    edge_attr=edge_attr_targets
                )  # (b, n, n)

                # First forward pass
                edge_logits, node_logits, edge_mask, node_mask, posterior = model(batch)
                edge_loss = F.cross_entropy(edge_logits[edge_mask], edge_targets[edge_mask])
                if cfg.dataset.num_node_types > 1:
                    node_loss = F.cross_entropy(node_logits, node_attr_targets[:, 0])
                    reconstruction_loss = edge_loss + node_loss
                else:
                    reconstruction_loss = edge_loss
                    node_loss = 0.

                kl_loss = posterior.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                loss = reconstruction_loss + cfg.kl_weight * kl_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if master_process:
                loss_window.append([
                    float(loss.detach().cpu()),
                    float(edge_loss.detach().cpu()),
                    float(node_loss.detach().cpu()) if type(node_loss) is not float else node_loss
                ])

            # Logging 
            if step % int(cfg.log_after) == 0 and master_process:
                if cfg.wandb_project is not None and master_process:
                    agg_loss = sum([losses[0] for losses in loss_window]) / len(loss_window)
                    agg_edge_loss = sum([losses[1] for losses in loss_window]) / len(loss_window)
                    agg_node_loss = sum([losses[2] for losses in loss_window]) / len(loss_window)
                    wandb.log(
                        dict(
                            train_loss=agg_loss,
                            edge_loss=agg_edge_loss,
                            node_loss=agg_node_loss,
                            lr=optimizer.param_groups[0]["lr"],
                            time=time.time() - start_time,
                        ),
                        step=step,
                    )
                loss_window = []
                start_time = time.time()

            # Validation
            if step % int(cfg.val_after) == 0:
                logger.info(f"Completed epoch [{device_id}]")
                if device_count > 1:
                    epoch += 1
                    train_loader.sampler.set_epoch(epoch)
                if master_process:
                    logger.info("Evaluating model")
                    module = model.module if device_count > 1 else model
                    module.eval()
                    sample_acc, val_edge_metrics, val_node_metrics = evaluate(
                        module,
                        val_loader,
                        device_id,
                        cfg.dataset.num_node_types,
                        cfg.dataset.num_edge_types,
                        cfg.dataset.num_nodes
                    )
                    logger.info(f"Evaluation done.")
                    # Checkpointing
                    if best_sample_acc is None or sample_acc > best_sample_acc:
                        best_sample_acc = sample_acc
                        best_edge_metrics = val_edge_metrics
                        best_node_metrics = val_node_metrics
                        if cfg.checkpoint is not None:
                            module = model.module if device_count > 1 else model
                            save_checkpoint(
                                checkpoint_file,
                                step,
                                module,
                                optimizer,
                                best_sample_acc=best_sample_acc,
                                best_edge_metrics=best_edge_metrics,
                                best_node_metrics=best_node_metrics
                            )
                    # Wandb logging
                    if cfg.wandb_project is not None and master_process:
                        wandb.log(
                            dict(
                                sample_acc=sample_acc,
                                best_sample_acc=best_sample_acc,
                                val_edge_metrics=val_edge_metrics,
                                val_node_metrics=val_node_metrics,
                            ),
                            step=step,
                        )
                if device_count > 1:
                    torch.distributed.barrier()
                model.train()

            step += 1

    logger.info(f"Training complete 🥳")

    if master_process:
        wandb.finish()

    if device_count > 1:
        destroy_process_group()


if __name__ == "__main__":
    main()
