import hydra
import torch

torch.autograd.set_detect_anomaly(True)
from loguru import logger

from evaluation.autoencoder import evaluate_autoencoder
from fm import compute_n_nodes_distr
from fm.fm_helpers import build_autoencoder
from utils import (
    setup_everything,
    create_dataloaders,
    instantiate_dataset,
    load_checkpoint
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

    logger.info(f"Length of test loader : {len(test_dataset)}")

    model = build_autoencoder(cfg, device_id)

    model.eval()
    load_checkpoint(
        f"{ckpt_dir}/{cfg.checkpoint}.pt", model, device_id
    )
    for param in model.parameters():
        param.requires_grad = False

    test_sample_acc, test_edge_metrics, test_node_metrics, test_sample_accuracy_by_num_nodes = evaluate_autoencoder(
        model,
        test_loader,
        device_id,
        cfg.dataset.num_node_types,
        cfg.dataset.num_edge_types,
        cfg.dataset.num_nodes,
        pad_edges_to_max_num_nodes=True,
        disable_metric_sync=False,
    )
    logger.info(f"Test performance : sample accuracy {test_sample_acc} - "
                f"edge metrics : {test_edge_metrics} - "
                f"node metrics : {test_node_metrics} - "
                f"sample accuracy by num nodes : {test_sample_accuracy_by_num_nodes}")


if __name__ == "__main__":
    main()
