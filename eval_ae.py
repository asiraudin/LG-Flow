import hydra
import torch

torch.autograd.set_detect_anomaly(True)
from loguru import logger

from utils import (
    setup_everything,
    save_checkpoint,
    create_dataloaders,
    setup_training,
    instantiate_dataset,
    load_checkpoint
)
from models import AutoencoderKL, EncoderLPE, EncodermLPE, DecoderUnDirected, DecoderDirected
from fm import compute_n_nodes_distr
from torch_geometric.utils import to_dense_adj, to_dense_batch

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall
)


@torch.no_grad()
def evaluate(model, loader, device, num_node_types, num_edge_types, max_num_nodes):
    model.eval()
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
            edge_attr=edge_attr_targets,
            max_num_nodes=max_num_nodes
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

    return sample_accuracy, 0., edge_metrics, node_metrics


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

    logger.info(f"Length of test loader : {len(test_dataset)}")

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

    model.eval()
    load_checkpoint(
        f"{ckpt_dir}/{cfg.checkpoint}.pt", model, device_id
    )
    for param in model.parameters():
        param.requires_grad = False

    test_sample_acc, test_acc, test_edge_metrics, test_node_metrics = evaluate(
        model,
        test_loader,
        device_id,
        cfg.dataset.num_node_types,
        cfg.dataset.num_edge_types,
        cfg.dataset.num_nodes
    )
    logger.info(f"Test performance : sample accuracy {test_sample_acc} - "
                f"edge metrics : {test_edge_metrics} - "
                f"node metrics : {test_node_metrics}")


if __name__ == "__main__":
    main()
