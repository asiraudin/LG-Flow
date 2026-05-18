from typing import Any

import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


def _disable_sync(metric: Any) -> None:
    if metric is not None:
        if hasattr(metric, "sync_on_compute"):
            metric.sync_on_compute = False
        if hasattr(metric, "dist_sync_on_step"):
            metric.dist_sync_on_step = False


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(numerator, dtype=torch.float32)                                 # (c,)
    valid = denominator > 0                                                                   # (c,)
    result[valid] = numerator[valid].float() / denominator[valid].float()                     # (c,)
    return result


def _update_one_vs_rest_counts(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
) -> None:
    for label_idx in range(tp.shape[0]):
        pred_label = (pred == label_idx) & valid_mask                                         # (...)
        target_label = (target == label_idx) & valid_mask                                     # (...)
        tp[label_idx] += (pred_label & target_label).sum()                                    # ()
        fp[label_idx] += (pred_label & ~target_label).sum()                                   # ()
        fn[label_idx] += (~pred_label & target_label).sum()                                   # ()


def _compute_per_label_metrics(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    total_valid: int,
) -> dict[str, torch.Tensor]:
    tn = torch.full_like(tp, fill_value=total_valid) - tp - fp - fn                           # (c,)
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)                                       # (c,)
    precision = _safe_divide(tp, tp + fp)                                                     # (c,)
    recall = _safe_divide(tp, tp + fn)                                                        # (c,)
    f1 = _safe_divide(2 * precision * recall, precision + recall)                             # (c,)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _finalize_sample_accuracy_by_num_nodes(
    correct_by_num_nodes: dict[int, int],
    total_by_num_nodes: dict[int, int],
) -> dict[str, torch.Tensor]:
    sorted_num_nodes = sorted(total_by_num_nodes.keys())                                       # [k]
    if len(sorted_num_nodes) == 0:
        return {
            "num_nodes": torch.empty((0,), dtype=torch.long),
            "sample_accuracy": torch.empty((0,), dtype=torch.float32),
            "num_samples": torch.empty((0,), dtype=torch.long),
        }

    num_nodes_tensor = torch.tensor(sorted_num_nodes, dtype=torch.long)                        # (k,)
    num_samples_tensor = torch.tensor(
        [total_by_num_nodes[num_nodes] for num_nodes in sorted_num_nodes],
        dtype=torch.long,
    )                                                                                          # (k,)
    correct_tensor = torch.tensor(
        [correct_by_num_nodes.get(num_nodes, 0) for num_nodes in sorted_num_nodes],
        dtype=torch.float32,
    )                                                                                          # (k,)
    sample_accuracy_tensor = correct_tensor / num_samples_tensor.float()                       # (k,)
    return {
        "num_nodes": num_nodes_tensor,
        "sample_accuracy": sample_accuracy_tensor,
        "num_samples": num_samples_tensor,
    }


@torch.no_grad()
def evaluate_autoencoder(
    model: torch.nn.Module,
    loader: Any,
    device: Any,
    num_node_types: int,
    num_edge_types: int,
    max_num_nodes: int,
    *,
    pad_edges_to_max_num_nodes: bool,
    disable_metric_sync: bool,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor | dict[str, torch.Tensor]],
    dict[str, torch.Tensor | dict[str, torch.Tensor] | None],
    dict[str, torch.Tensor],
]:
    model.eval()
    edge_metrics = {
        "accuracy": MulticlassAccuracy(num_classes=num_edge_types + 1, ignore_index=-1).to(device),
        "precision": MulticlassPrecision(num_classes=num_edge_types + 1, ignore_index=-1).to(device),
        "recall": MulticlassRecall(num_classes=num_edge_types + 1, ignore_index=-1).to(device),
        "f1": MulticlassF1Score(num_classes=num_edge_types + 1, ignore_index=-1).to(device),
        "sample_accuracy": MulticlassAccuracy(
            num_classes=num_edge_types + 1,
            ignore_index=-1,
            multidim_average="samplewise",
        ).to(device),
    }
    num_edge_classes = num_edge_types + 1
    edge_tp = torch.zeros((num_edge_classes,), dtype=torch.long, device=device)               # (c,)
    edge_fp = torch.zeros((num_edge_classes,), dtype=torch.long, device=device)               # (c,)
    edge_fn = torch.zeros((num_edge_classes,), dtype=torch.long, device=device)               # (c,)
    total_valid_edge = 0
    edge_sample_correct_total = 0

    if num_node_types > 1:
        node_metrics = {
            "accuracy": MulticlassAccuracy(num_classes=num_node_types, ignore_index=-1).to(device),
            "precision": MulticlassPrecision(num_classes=num_node_types, ignore_index=-1).to(device),
            "recall": MulticlassRecall(num_classes=num_node_types, ignore_index=-1).to(device),
            "f1": MulticlassF1Score(num_classes=num_node_types, ignore_index=-1).to(device),
            "sample_accuracy": MulticlassAccuracy(
                num_classes=num_node_types,
                ignore_index=-1,
                multidim_average="samplewise",
            ).to(device),
        }
        node_tp = torch.zeros((num_node_types,), dtype=torch.long, device=device)             # (c,)
        node_fp = torch.zeros((num_node_types,), dtype=torch.long, device=device)             # (c,)
        node_fn = torch.zeros((num_node_types,), dtype=torch.long, device=device)             # (c,)
        total_valid_node = 0
        node_sample_correct_total = 0
    else:
        node_metrics = {key: None for key in edge_metrics.keys()}
        node_tp = node_fp = node_fn = None
        total_valid_node = 0
        node_sample_correct_total = 0

    sample_correct_by_num_nodes: dict[int, int] = {}
    sample_total_by_num_nodes: dict[int, int] = {}
    total_num_samples = 0

    if disable_metric_sync:
        for metric in edge_metrics.values():
            _disable_sync(metric)
        for metric in node_metrics.values():
            _disable_sync(metric)

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            edge_attr_targets = batch.edge_attr[..., 0] + 1
            node_attr_targets = batch.x[..., 0]
            e_hat, x_hat, edge_mask, _, _ = model(batch, sample_posterior=False)

            e_hat = e_hat.softmax(dim=-1).argmax(dim=-1)  # (bs, n, n)
            x_hat = x_hat.softmax(dim=-1).argmax(dim=-1)  # (n_total,)

            dense_adj_kwargs = dict(
                edge_index=batch.edge_index,
                batch=batch.batch,
                edge_attr=edge_attr_targets,
            )
            if pad_edges_to_max_num_nodes:
                dense_adj_kwargs["max_num_nodes"] = max_num_nodes
            e = to_dense_adj(**dense_adj_kwargs)
            e = e.masked_fill(edge_mask == 0, -1)

            x, _ = to_dense_batch(node_attr_targets, batch.batch, fill_value=-1, max_num_nodes=max_num_nodes)
            x_hat, _ = to_dense_batch(x_hat, batch.batch, max_num_nodes=max_num_nodes)

            valid_edge_mask = e != -1                                                         # (bs, n, n)
            total_valid_edge += int(valid_edge_mask.sum().item())                             # ()
            _update_one_vs_rest_counts(e_hat, e, valid_edge_mask, edge_tp, edge_fp, edge_fn)

            edge_sample_correct = (
                ((e_hat == e) | ~valid_edge_mask).reshape(e.shape[0], -1).all(dim=-1)
            )                                                                                 # (bs,)
            edge_sample_correct_total += int(edge_sample_correct.sum().item())                # ()

            n_nodes = torch.bincount(batch.batch, minlength=e.shape[0]).long()                # (bs,)
            if num_node_types > 1:
                valid_node_mask = x != -1                                                     # (bs, n)
                total_valid_node += int(valid_node_mask.sum().item())                         # ()
                _update_one_vs_rest_counts(x_hat, x, valid_node_mask, node_tp, node_fp, node_fn)

                node_sample_correct = (
                    ((x_hat == x) | ~valid_node_mask).reshape(x.shape[0], -1).all(dim=-1)
                )                                                                             # (bs,)
                node_sample_correct_total += int(node_sample_correct.sum().item())            # ()
                sample_correct = edge_sample_correct & node_sample_correct                     # (bs,)
            else:
                sample_correct = edge_sample_correct                                           # (bs,)

            total_num_samples += sample_correct.shape[0]                                      # ()
            for sample_idx in range(sample_correct.shape[0]):
                num_nodes_i = int(n_nodes[sample_idx].item())                                 # ()
                sample_total_by_num_nodes[num_nodes_i] = sample_total_by_num_nodes.get(num_nodes_i, 0) + 1
                sample_correct_by_num_nodes[num_nodes_i] = (
                    sample_correct_by_num_nodes.get(num_nodes_i, 0)
                    + int(sample_correct[sample_idx].item())
                )

            for edge_metric, node_metric in zip(edge_metrics.values(), node_metrics.values()):
                edge_metric(e_hat, e)
                if num_node_types > 1:
                    node_metric(x_hat, x)

    edge_metrics = {key: value.compute() for key, value in edge_metrics.items()}
    edge_metrics["per_label"] = _compute_per_label_metrics(edge_tp, edge_fp, edge_fn, total_valid_edge)
    if num_node_types > 1:
        node_metrics = {key: value.compute() for key, value in node_metrics.items()}
        node_metrics["per_label"] = _compute_per_label_metrics(node_tp, node_fp, node_fn, total_valid_node)

    edge_sample_accuracy = torch.isclose(
        edge_metrics["sample_accuracy"],
        torch.ones_like(edge_metrics["sample_accuracy"]),
        rtol=1e-6,
    )
    edge_metrics["sample_accuracy"] = torch.tensor(
        edge_sample_correct_total / max(total_num_samples, 1),
        dtype=torch.float32,
        device=device,
    )

    if num_node_types > 1:
        node_sample_accuracy = torch.isclose(
            node_metrics["sample_accuracy"],
            torch.ones_like(node_metrics["sample_accuracy"]),
            rtol=1e-6,
        )
        node_metrics["sample_accuracy"] = torch.tensor(
            node_sample_correct_total / max(total_num_samples, 1),
            dtype=torch.float32,
            device=device,
        )
        sample_accuracy = torch.tensor(
            sum(sample_correct_by_num_nodes.values()) / max(total_num_samples, 1),
            dtype=torch.float32,
            device=device,
        )
    else:
        sample_accuracy = edge_metrics["sample_accuracy"]

    sample_accuracy_by_num_nodes = _finalize_sample_accuracy_by_num_nodes(
        sample_correct_by_num_nodes,
        sample_total_by_num_nodes,
    )
    return sample_accuracy, edge_metrics, node_metrics, sample_accuracy_by_num_nodes
