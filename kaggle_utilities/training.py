"""Training utilities: device setup, LR schedule, AMP, loss helpers."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn


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
        if output["loss"].dim() == 0:
            output["loss"] = output["loss"].unsqueeze(0)
        return output


def set_up_device(model: nn.Module):
    """
    Set up the best available device and wrap model for multi-GPU.

    Returns (model, device) where model may be DataParallel-wrapped
    and is moved to the appropriate device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPU(s): "
              + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)))
        if n_gpus > 1:
            model = nn.DataParallel(LossUnsqueezeWrapper(model))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    model = model.to(device)
    return model, device


def cosine_lr(step: int, max_steps: int, learning_rate: float,
              warmup_steps: int = 500) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Returns the learning rate for the given step.
    """
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def get_amp_context(device: torch.device):
    """
    Return the appropriate AMP autocast context manager.

    torch.autocast('cuda', dtype=torch.float16) on GPU,
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
