import hydra
import time
import torch
import torch.nn.functional as F
from loguru import logger
import wandb
from typing import Any

from evaluation.autoencoder import evaluate_autoencoder
from fm import compute_n_nodes_distr
from fm.fm_helpers import build_autoencoder
from utils import (
    setup_everything,
    save_checkpoint,
    create_dataloaders,
    setup_training,
    instantiate_dataset
)
from torch_geometric.utils import to_dense_adj, to_dense_batch

from torch.distributed import destroy_process_group


def _flatten_metric_dict(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_metric_dict(full_key, value))
            continue

        if torch.is_tensor(value):
            if value.ndim == 0:
                flattened[full_key] = float(value.detach().cpu())
            elif value.ndim == 1:
                for idx in range(value.shape[0]):
                    flattened[f"{full_key}/label_{idx}"] = float(value[idx].detach().cpu())
        elif value is not None:
            flattened[full_key] = value
    return flattened


def _build_sample_accuracy_bar_plot(sample_accuracy_by_num_nodes: dict[str, torch.Tensor]) -> Any | None:
    num_nodes = sample_accuracy_by_num_nodes["num_nodes"].tolist()
    sample_accuracy = sample_accuracy_by_num_nodes["sample_accuracy"].tolist()
    num_samples = sample_accuracy_by_num_nodes["num_samples"].tolist()
    if len(num_nodes) == 0:
        return None

    if len(num_nodes) <= 20:
        labels = [str(num_node) for num_node in num_nodes]
        accuracies = sample_accuracy
        counts = num_samples
    else:
        num_bins = min(10, len(num_nodes))
        min_nodes = min(num_nodes)
        max_nodes = max(num_nodes)
        bin_edges = torch.linspace(float(min_nodes), float(max_nodes), steps=num_bins + 1)
        labels = []
        accuracies = []
        counts = []
        for bin_idx in range(num_bins):
            left = int(bin_edges[bin_idx].item())
            right = int(bin_edges[bin_idx + 1].item())
            if bin_idx == num_bins - 1:
                selected = [idx for idx, num_node in enumerate(num_nodes) if left <= num_node <= right]
            else:
                selected = [idx for idx, num_node in enumerate(num_nodes) if left <= num_node < right]
            if len(selected) == 0:
                continue

            total_count = sum(num_samples[idx] for idx in selected)
            weighted_accuracy = sum(sample_accuracy[idx] * num_samples[idx] for idx in selected) / total_count
            labels.append(f"{left}-{right}")
            accuracies.append(weighted_accuracy)
            counts.append(total_count)

    table = wandb.Table(
        data=[[label, accuracy, count] for label, accuracy, count in zip(labels, accuracies, counts)],
        columns=["node_count_bin", "sample_accuracy", "num_samples"],
    )
    return wandb.plot.bar(
        table,
        "node_count_bin",
        "sample_accuracy",
        title="Sample Accuracy by Node Count",
    )


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

    model = build_autoencoder(cfg, device_id)

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
                    sample_acc, val_edge_metrics, val_node_metrics, val_sample_accuracy_by_num_nodes = evaluate_autoencoder(
                        module,
                        val_loader,
                        device_id,
                        cfg.dataset.num_node_types,
                        cfg.dataset.num_edge_types,
                        cfg.dataset.num_nodes,
                        pad_edges_to_max_num_nodes=False,
                        disable_metric_sync=True,
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
                        log_payload = dict(
                            sample_acc=float(sample_acc.detach().cpu()),
                            best_sample_acc=float(best_sample_acc.detach().cpu()) if torch.is_tensor(best_sample_acc) else best_sample_acc,
                        )
                        log_payload.update(_flatten_metric_dict("val_edge_metrics", val_edge_metrics))
                        log_payload.update(_flatten_metric_dict("val_node_metrics", val_node_metrics))
                        sample_accuracy_bar = _build_sample_accuracy_bar_plot(val_sample_accuracy_by_num_nodes)
                        if sample_accuracy_bar is not None:
                            log_payload["val_sample_accuracy_by_num_nodes"] = sample_accuracy_bar
                        wandb.log(log_payload, step=step)
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
