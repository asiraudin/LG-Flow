import random
import time
from pathlib import Path
from typing import Any

import hydra
import torch
import wandb
from loguru import logger
from torch.distributed import destroy_process_group
from torch_geometric.utils import to_dense_batch

from fm import compute_n_nodes_distr
from fm.fm_helpers import (
    build_autoencoder,
    build_denoiser,
    build_interpolant,
    build_sampling_metrics,
    evaluate_samples,
    generate_samples,
    get_selection_metric,
    get_selection_mode,
    should_save_every_eval,
)
from utils import (
    create_dataloaders,
    instantiate_dataset,
    load_checkpoint,
    save_checkpoint,
    setup_everything,
    setup_training,
)


@hydra.main(version_base=None, config_path="./configs", config_name="planar_fm_train")
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

    sampling_metrics = None
    if master_process:
        sampling_metrics = build_sampling_metrics(cfg, train_loader, val_loader, test_loader, test=False)

    autoencoder = build_autoencoder(cfg, device_id)
    autoencoder.eval()
    load_checkpoint(f"{ckpt_dir}/{cfg.ae_checkpoint_file}.pt", autoencoder, device_id)
    for param in autoencoder.parameters():
        param.requires_grad = False

    interpolant = build_interpolant(cfg, device_id)
    model = build_denoiser(cfg, device_id)
    model, ema_model, optimizer, lr_scheduler, step, checkpoint_file, best_val = setup_training(
        cfg=cfg,
        model=model,
        master_process=master_process,
        device=device,
        device_id=device_id,
        device_count=device_count,
        dtype=dtype,
        ckpt_dir=ckpt_dir,
    )
    best_value = best_val[0]
    selection_metric = get_selection_metric(cfg)
    selection_mode = get_selection_mode(cfg)
    save_every_eval = should_save_every_eval(cfg)
    logger.info(f"Starting/resuming training for {int(cfg.num_steps) - (step - 1)} steps 🚀")

    if master_process:
        start_time = time.time()
    if device_count > 1:
        train_loader.sampler.set_epoch(epoch := 0)

    first_batch = True
    q: float | None = None
    scale_factor = 1.0
    while step <= cfg.num_steps:
        model.train()
        loss_window: list[float] = []
        for batch in train_loader:
            batch = batch.to(device_id)
            if first_batch:
                posterior = autoencoder.encode(batch)
                z = posterior.sample().detach() if cfg.get("sample", True) else posterior.mode().detach()
                scale_factor = 1.0 / z.flatten().std()
                if cfg.dataset.directed:
                    q = batch.q.squeeze()[0].item()
                first_batch = False

            with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
                posterior = autoencoder.encode(batch)
                x_1 = posterior.sample().detach() if cfg.get("sample", True) else posterior.mode().detach()
                x_1 = scale_factor * x_1  # (n_total, pe_dim)
                x_1, mask = to_dense_batch(x_1, batch.batch)  # (bs, n, pe_dim), (bs, n)
                x_t, t, x_0 = interpolant.corrupt_batch(x_1, mask)

                if random.random() < cfg.sc_prob:
                    with torch.no_grad():
                        x_sc = model(x_t, t, mask)
                else:
                    x_sc = None

                out = model(x_t, t, mask, x_sc)  # (bs, n, pe_dim)
                if cfg.param == "x_0":
                    loss = interpolant.criterion(x_0, t, out, mask)
                elif cfg.param == "x_1":
                    loss = interpolant.criterion(x_1, t, out, mask)
                else:
                    loss = interpolant.criterion(x_1 - x_0, t, out, mask)

            loss.backward()
            if cfg.gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            if master_process:
                ema_model.update()
                loss_window.append(float(loss.detach().cpu()))

            if step % int(cfg.log_after) == 0 and master_process:
                if cfg.wandb_project is not None:
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
                        step=step,
                        sampling_metrics=sampling_metrics,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                    )
                    current_value = metrics[selection_metric]
                    if best_value is None or (
                        current_value > best_value if selection_mode == "max" else current_value < best_value
                    ):
                        best_value = current_value
                        if cfg.checkpoint is not None and not save_every_eval:
                            save_checkpoint(
                                checkpoint_file,
                                step,
                                ema_model,
                                optimizer,
                                ema=True,
                                best_val=best_value,
                            )
                    if cfg.checkpoint is not None and save_every_eval:
                        checkpoint_path = Path(checkpoint_file)
                        save_path = str(
                            checkpoint_path.with_name(
                                f"{checkpoint_path.stem}{f'step_{step}'}{checkpoint_path.suffix}"
                            )
                        )
                        save_checkpoint(
                            save_path,
                            step,
                            ema_model,
                            optimizer,
                            ema=True,
                            best_val=best_value,
                        )
                    ema_model.train()
                if device_count > 1 and cfg.dataset.directed:
                    torch.distributed.barrier()

            step += 1


if __name__ == "__main__":
    main()
