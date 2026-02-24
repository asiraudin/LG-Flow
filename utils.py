import torch
from loguru import logger
import inspect
import os, datetime
import math

import wandb
import torch.distributed as dist
from torch.distributed import init_process_group
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import (
    PlanarDataset, TreeDataset, EgoDataset, ProteinDataset, PointCloudsDataset, MOSESDataset, GuacamolDataset, 
    TPUGraphDataset, ERDAGDataset, PriceDataset, Transform, OrbitTransform
)

from ema_pytorch import EMA
from torch_geometric.transforms import Compose

DATASETS = {'planar': PlanarDataset, 'moses': MOSESDataset, 'guacamol': GuacamolDataset,
            'ego': EgoDataset, 'protein': ProteinDataset, 'tree': TreeDataset,
            'tpu_tile': TPUGraphDataset, 'point_clouds': PointCloudsDataset,
            'er_dag': ERDAGDataset, 'price': PriceDataset}


def configure_optimizers(
    model: torch.nn.Module,
    weight_decay,
    learning_rate,
    betas,
    device_type,
):
    """Adapted from https://github.com/karpathy/nanoGPT"""
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [
        p
        for n, p in param_dict.items()
        if p.dim() >= 2  # and (n.startswith("tokenizer.") or n.startswith("mlp."))
    ]
    nodecay_params = [
        p
        for n, p in param_dict.items()
        if p.dim() < 2  # and (n.startswith("tokenizer.") or n.startswith("mlp."))
    ]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"using fused AdamW: {use_fused}")

    return optimizer


def count_parameters(model: torch.nn.Module):
    """Source: https://stackoverflow.com/a/62508086"""
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        # table.add_row([name, params])
        total_params += params
    # logger.info(f"\n{str(table)}")
    return total_params


def ensure_root_folder(root, master_process=True):
    if not os.path.exists(root) and master_process:
        logger.info(f"Creating root directory {root}")
        os.makedirs(root)

    if not os.path.exists(data_dir := f"{root}/data") and master_process:
        logger.info(f"Creating data directory {data_dir}")
        os.makedirs(data_dir)

    if not os.path.exists(ckpt_dir := f"{root}/ckpt") and master_process:
        logger.info(f"Creating ckpt directory {ckpt_dir}")
        os.makedirs(ckpt_dir)

    return data_dir, ckpt_dir


def load_checkpoint(checkpoint_file, model, device_id=None, ema=False):
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading pre-trained checkpoint from {checkpoint_file}")
        load_args = (
            dict(map_location=f"cuda:{device_id}") if torch.cuda.is_available() else {}
        )
        checkpoint = torch.load(checkpoint_file, **load_args)
        if not ema:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["ema_model_state_dict"])
    else:
        raise ValueError(f"Could not find checkpoint {checkpoint_file}, please provide valid autoencoder checkpoint")


def save_checkpoint(checkpoint_file, step, model, optimizer, ema=False,
                    **kwargs):
    logger.info(f"Creating and saving checkpoint to {checkpoint_file}")
    if not ema:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                **kwargs
            },
            checkpoint_file,
        )
    else:
        torch.save(
            {
                "ema_model_state_dict": model.state_dict(),
                "model_state_dict": model.online_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                **kwargs
            },
            checkpoint_file,
        )


def continue_from_checkpoint(checkpoint_file, model, optimizer, master_process, device_id=None, autoencoder=False,
                             ema_model=None):
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading pre-trained checkpoint from {checkpoint_file}")
        load_args = (
            dict(map_location=f"cuda:{device_id}") if torch.cuda.is_available() else {}
        )
        if autoencoder:
            checkpoint = torch.load(checkpoint_file, **load_args)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            return (
                checkpoint["step"] + 1,
                checkpoint["best_sample_acc"],
                checkpoint["best_edge_metrics"],
                checkpoint["best_node_metrics"],
                None
            )
        else:
            checkpoint = torch.load(checkpoint_file, **load_args)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            ema_model = None
            if master_process:
                ema_model = EMA(model, beta=0.9999, update_after_step=1000, update_every=1,
                                allow_different_devices=True)
                ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            return checkpoint["step"], checkpoint["best_val"], ema_model
    else:
        logger.info(
            f"Could not find checkpoint {checkpoint_file}, starting training from scratch"
        )
        if autoencoder:
            return 1, None, None, None, None
        else:
            ema_model = None
            if master_process:
                ema_model = EMA(model, beta=0.9999, update_after_step=1000, update_every=1,
                                allow_different_devices=True)
            return 1, None, ema_model


class ConstantLRScheduler:
    """
    A simple constant learning rate scheduler.
    """
    def __init__(self, optimizer, lr: float):
        self.optimizer = optimizer
        self.lr = lr
        self.step()

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr


class CosineWithWarmupLR:
    """Adapted from https://github.com/karpathy/nanoGPT"""

    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        lr: float,
        lr_decay_iters: int,
        min_lr: float,
        epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.epoch = epoch
        self.step()

    def step(self):
        self.epoch += 1
        lr = self._get_lr(self.epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, epoch: int):
        # 1) linear warmup for warmup_iters steps
        if epoch < self.warmup_iters:
            return self.lr * epoch / self.warmup_iters
        # 2) if epoch > lr_decay_iters, return min learning rate
        if epoch > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (epoch - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.lr - self.min_lr)


def ddp_setup():
    if torch.cuda.device_count() > 1:
        init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def accelerator_setup():
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_id = int(os.environ["LOCAL_RANK"])
            master_process = device_id == 0
        else:
            device_id = 0
            master_process = True
    else:
        device = "cpu"
        device_id = "cpu"
        device_count = 1
        master_process = True

    return device, device_id, device_count, master_process


def setup_everything(cfg):
    ddp_setup()

    device, device_id, device_count, master_process = accelerator_setup()
    logger.info(f"Accelerator: {device}, num. devices {device_count}")

    data_dir, ckpt_dir = ensure_root_folder(cfg.root, master_process)

    if cfg.wandb_project is not None and master_process:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config=dict(cfg),
        )

    torch.set_float32_matmul_precision("medium")
    logger.info(f"Setting float32 matmul precision to medium")

    # dtype = (
    #     "bfloat16"
    #     if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    #     else "float16"
    # )
    # logger.info(f"Data type: {dtype}")
    # tdtype = torch.float16 if dtype == "float16" else torch.bfloat16
    dtype = 'float32'
    tdtype = torch.float32
    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    return device, device_id, device_count, master_process, data_dir, ckpt_dir, dtype, tdtype


def create_dataloaders(train_dataset, val_dataset, test_dataset, cfg, device_count, master_process):
    if device_count > 1:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size // device_count,
            num_workers=cfg.num_workers,
            shuffle=False,
            sampler=DistributedSampler(train_dataset),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
        )

    if master_process:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
    if master_process:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, None, None


def setup_training(cfg, model, master_process, device, device_id, device_count, dtype, ckpt_dir, autoencoder=False,
                   ):
    if master_process:
        num_params = count_parameters(model)

        if cfg.wandb_project is not None and master_process:
            wandb.log(dict(num_params=num_params))

        # logger.info(model)
        logger.info(f"Number of parameters: {num_params}")

    optimizer = configure_optimizers(
        model,
        cfg.weight_decay,
        cfg.lr,
        (0.9, 0.95),
        device,
    )

    if cfg.checkpoint is not None:
        checkpoint_file = f"{ckpt_dir}/{cfg.checkpoint}.pt"
        logger.info(f"Trying to continue from checkpoint {checkpoint_file}")
        step, *best, ema_model = continue_from_checkpoint(
            checkpoint_file, model, optimizer, master_process, device_id, autoencoder=autoencoder
        )
    else:
        checkpoint_file = None
        ema_model = None
        if master_process and not autoencoder:
            ema_model = EMA(model, ema_model=ema_model, beta=0.9999, update_after_step=1000, update_every=1,
                            allow_different_devices=True)
        step, best = (1, [None, None, None, None]) if autoencoder else (1, None)

    if cfg.lr_scheduler == 'cosine':
        lr_scheduler = CosineWithWarmupLR(
            optimizer, cfg.num_warmup_steps, cfg.lr, cfg.lr_decay_iters, cfg.min_lr, step - 1
        )
    elif cfg.lr_scheduler == 'constant':
        lr_scheduler = ConstantLRScheduler(
            optimizer, cfg.lr
        )
    else:
        raise ValueError(f"Incorrect value for lr_scheduler: {cfg.lr_scheduler}")

    if master_process:
        logger.info(f"Optimizer + scheduler with lr {cfg.lr} ready")

    if device_count > 1:
        logger.info("Creating DDP module")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device_id])

    return model, ema_model, optimizer, lr_scheduler, step, checkpoint_file, best


def instantiate_dataset(name, data_dir, cfg, master_process):
    laplacian_transform = Transform(
        directed=cfg.dataset.directed, 
        normalized_laplacian=cfg.encoder.normalized_laplacian, 
        normalize_eigenvecs=cfg.encoder.normalize_eigenvecs,
        large_graph=cfg.dataset.large_graph
    )

    pre_transform = Compose([laplacian_transform, OrbitTransform()])
    try:
        dataset = DATASETS[name]
    except KeyError:
        raise ValueError(f"Dataset {name} not found. Available datasets: {list(DATASETS.keys())}")

    if master_process:
        _ = dataset(data_dir, split='train', pre_transform=pre_transform)
        _ = dataset(data_dir, split='val', pre_transform=pre_transform)
        _ = dataset(data_dir, split='test', pre_transform=pre_transform)
    
    if dist.is_initialized():
        dist.barrier()

    train_dataset = dataset(data_dir, split='train', pre_transform=pre_transform)
    val_dataset = dataset(data_dir, split='val', pre_transform=pre_transform)
    test_dataset = dataset(data_dir, split='test', pre_transform=pre_transform)
    return train_dataset, val_dataset, test_dataset
