from pathlib import Path
from typing import Any

import hydra
from loguru import logger

from fm import compute_n_nodes_distr
from fm.fm_helpers import (
    build_autoencoder,
    build_ema_model,
    build_interpolant,
    build_sampling_metrics,
    evaluate_samples,
    generate_samples,
)
from utils import create_dataloaders, instantiate_dataset, load_checkpoint, setup_everything


@hydra.main(version_base=None, config_path="./configs", config_name="planar_fm_test")
def main(cfg: Any) -> None:
    device, device_id, device_count, master_process, data_dir, ckpt_dir, dtype, tdtype = setup_everything(cfg)

    logger.info(f"Loading datasets from {data_dir}")
    train_dataset, val_dataset, test_dataset = instantiate_dataset(
        name=cfg.dataset.name,
        data_dir=data_dir,
        cfg=cfg,
        master_process=master_process,
    )
    node_distribution = compute_n_nodes_distr(
        train_n_nodes=train_dataset.num_nodes,
        val_n_nodes=val_dataset.num_nodes,
        test_n_nodes=test_dataset.num_nodes,
    )
    logger.info("Dataset loaded")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        cfg=cfg,
        device_count=device_count,
        master_process=master_process,
    )
    sampling_metrics = build_sampling_metrics(cfg, train_loader, val_loader, test_loader, test=True) if master_process else None

    autoencoder = build_autoencoder(cfg, device_id)
    autoencoder.eval()
    load_checkpoint(f"{ckpt_dir}/{cfg.ae_checkpoint_file}.pt", autoencoder, device_id)
    for param in autoencoder.parameters():
        param.requires_grad = False

    interpolant = build_interpolant(cfg, device_id)
    ema_model = build_ema_model(cfg, device_id)
    load_checkpoint(f"{ckpt_dir}/{cfg.checkpoint}.pt", ema_model, device_id, ema=True)
    ema_model.eval()

    logger.info("Testing begins 🤞🏼")
    batch = next(iter(train_loader))
    batch = batch.to(device_id)
    posterior = autoencoder.encode(batch)
    z = posterior.sample().detach() if cfg.get("sample", True) else posterior.mode().detach()
    scale_factor = 1.0 / z.flatten().std()
    q = batch.q.squeeze()[0].item() if cfg.dataset.directed else None

    generated_samples = generate_samples(
        interpolant=interpolant,
        model=ema_model,
        autoencoder=autoencoder,
        scale_factor=scale_factor,
        node_distribution=node_distribution,
        cfg=cfg,
        device_id=device_id,
        atom_decoder=getattr(train_dataset, "atom_decoder", None),
        q=q,
    )
    metrics = evaluate_samples(
        generated_samples=generated_samples,
        cfg=cfg,
        device_id=device_id,
        step=0,
        sampling_metrics=sampling_metrics,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    if cfg.dataset.molecular:
        output_path = Path(cfg.root) / f"{cfg.checkpoint}_samples_{cfg.num_sampling_steps}.txt"
        output_path.write_text("\n".join(generated_samples), encoding="utf-8")

    logger.info(f"Test performance on {cfg.num_samples * cfg.num_sample_batch} samples: {metrics}")


if __name__ == "__main__":
    main()
