"""Training loop helpers: device setup, LR schedule, AMP, gradient accumulation."""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LossUnsqueezeWrapper(nn.Module):
    """
    Wraps a model so its forward() returns loss as a 1-element tensor
    instead of a scalar. This prevents the DataParallel "scalar gathering"
    warning without touching the model's own code.

    DataParallel gathers outputs from each GPU by concatenating along dim 0.
    Scalar tensors (0-dim) trigger a UserWarning. By unsqueezing to [1],
    DataParallel gathers to [n_gpus], which we .mean() later.

    The wrapped model's forward must return a dict with a "loss" key.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if output.get("loss") is not None:
            output["loss"] = output["loss"].unsqueeze(0)
        return output

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def setup_device(
    model: nn.Module,
    use_data_parallel: bool = True,
) -> tuple[nn.Module, torch.device]:
    """
    Move model to GPU(s) and optionally wrap in DataParallel.

    Handles three cases transparently:
    - CPU: model.to('cpu'), no wrapping
    - Single GPU: model.to('cuda'), no wrapping
    - Multi-GPU: wraps in LossUnsqueezeWrapper then DataParallel

    Args:
        model: The model to set up.
        use_data_parallel: If True and multiple GPUs available, use DataParallel.

    Returns:
        (model, device) tuple. Model may be wrapped in DataParallel.
    """
    if not torch.cuda.is_available():
        return model, torch.device("cpu")

    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()

    if n_gpus > 1 and use_data_parallel:
        model = LossUnsqueezeWrapper(model)
        model = nn.DataParallel(model)
        model = model.to(device)
        print(f"Using DataParallel on {n_gpus} GPUs")
    else:
        model = model.to(device)
        if n_gpus == 1:
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")

    return model, device


def cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 1e-5,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_amp_context(device: torch.device):
    """
    Return appropriate AMP autocast context manager.

    Returns torch.autocast('cuda', dtype=torch.float16) on GPU,
    nullcontext on CPU.
    """
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def get_grad_scaler(device: torch.device):
    """
    Return GradScaler for AMP on GPU, None on CPU.

    Uses torch.amp.GradScaler('cuda') (not deprecated torch.cuda.amp.GradScaler).
    """
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Return the underlying model, unwrapping DataParallel and/or
    LossUnsqueezeWrapper if present.

    Peels off layers in order:
    1. DataParallel -> .module
    2. LossUnsqueezeWrapper -> .model
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    if isinstance(model, LossUnsqueezeWrapper):
        model = model.model
    return model


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """
    Reduce loss tensor to a scalar, handling all three device paths:

    - Scalar (0-dim): single GPU or CPU -> return as-is
    - 1-dim tensor [n_gpus]: DataParallel gathered -> return .mean()
    """
    if loss.dim() == 0:
        return loss
    return loss.mean()


class TrainingContext:
    """
    Manages the training loop boilerplate: device setup, optimizer,
    LR schedule, AMP, gradient accumulation, logging, and checkpointing.

    Transparently handles CPU, single GPU, and multi-GPU (DataParallel).

    Usage:
        ctx = TrainingContext(
            model=model,
            max_steps=20000,
            learning_rate=3e-4,
            grad_accum_steps=16,
            checkpoint_dir="/kaggle/working/checkpoints",
        )
        for step, batch in ctx.training_steps(data_loader):
            loss = ctx.forward_backward(batch)
            ctx.step_optimizer()
            if ctx.should_log():
                ctx.log(loss)
            if ctx.should_save():
                ctx.save_checkpoint()
    """

    def __init__(
        self,
        model: nn.Module,
        max_steps: int = 20000,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 500,
        grad_accum_steps: int = 16,
        log_interval: int = 100,
        save_interval: int = 5000,
        checkpoint_dir: str = "/kaggle/working/checkpoints",
        use_data_parallel: bool = True,
    ):
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir

        # Device setup
        self.model, self.device = setup_device(model, use_data_parallel)

        # Optimizer (applied to the underlying model's parameters)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # AMP
        self.amp_context = get_amp_context(self.device)
        self.scaler = get_grad_scaler(self.device)

        # State
        self._step = 0
        self._micro_step = 0
        self._accumulated_loss = 0.0
        self._step_start_time = time.time()

        os.makedirs(checkpoint_dir, exist_ok=True)

    @property
    def step(self) -> int:
        return self._step

    def training_steps(self, data_loader: DataLoader):
        """
        Yield (step, batch) pairs from the data loader up to max_steps.

        Each yielded step corresponds to one gradient accumulation cycle.
        """
        data_iter = iter(data_loader)
        while self._step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)
            yield self._step, batch

    def forward_backward(self, batch: dict) -> float:
        """
        Run one micro-step: forward pass + scaled backward pass.

        Accumulates gradients over grad_accum_steps micro-steps.
        Returns the loss value for this micro-step.
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with self.amp_context:
            output = self.model(input_ids=input_ids, labels=labels)
            loss = reduce_loss(output["loss"])
            scaled_loss = loss / self.grad_accum_steps

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        loss_val = loss.item()
        self._accumulated_loss += loss_val
        self._micro_step += 1
        return loss_val

    def step_optimizer(self) -> bool:
        """
        Step the optimizer if we've accumulated enough micro-steps.

        Updates the learning rate, clips gradients, and resets accumulators.
        Returns True if an optimizer step was taken.
        """
        if self._micro_step < self.grad_accum_steps:
            return False

        # Update learning rate
        lr = cosine_lr(
            self._step, self.warmup_steps, self.max_steps,
            self.learning_rate,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Gradient clipping and optimizer step
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.optimizer.zero_grad()
        self._step += 1
        self._micro_step = 0
        self._accumulated_loss = 0.0
        self._step_start_time = time.time()
        return True

    def should_log(self) -> bool:
        return self._step > 0 and self._step % self.log_interval == 0

    def should_save(self) -> bool:
        return self._step > 0 and self._step % self.save_interval == 0

    def log(self, loss: float):
        """Print training metrics for the current step."""
        lr = self.optimizer.param_groups[0]["lr"]
        elapsed = time.time() - self._step_start_time
        print(
            f"step {self._step}/{self.max_steps} | "
            f"loss {loss:.4f} | lr {lr:.2e} | "
            f"time {elapsed:.1f}s"
        )

    def save_checkpoint(self, suffix: str | None = None):
        """Save model checkpoint."""
        name = suffix or f"step_{self._step}"
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        inner_model = unwrap_model(self.model)
        torch.save(inner_model.state_dict(), path)
        print(f"Saved checkpoint: {path}")
