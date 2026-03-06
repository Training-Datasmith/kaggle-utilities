"""TrainingContext: manages the training loop lifecycle."""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .training import (
    cosine_lr,
    get_amp_context,
    get_grad_scaler,
    reduce_loss,
    set_up_device,
    unwrap_model,
)


class TrainingContext:
    """
    Manages the training loop boilerplate: device setup, optimizer,
    LR schedule, AMP, gradient accumulation, logging, and checkpointing.
    Transparently handles CPU, single GPU, and multi-GPU (DataParallel).

    Usage:

        ctx = TrainingContext(
            model=model, max_steps=20000, learning_rate=3e-4,
            grad_accum_steps=16,
            checkpoint_dir="/kaggle/working/checkpoints",
        )
        ctx.load_checkpoint()   # resume from previous run if available

        for step, batch in ctx.training_steps(data_loader):
            loss = ctx.forward_backward(batch)
            if ctx.step_optimizer():
                losses.append(loss)
                if ctx.should_log():
                    ctx.log(loss)
                if ctx.should_save():
                    ctx.save_checkpoint()
    """

    RESUME_FILENAME = "resume.pt"

    def __init__(
        self,
        model: nn.Module,
        max_steps: int = 20000,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 500,
        grad_accum_steps: int = 16,
        log_interval: int = 100,
        save_interval: int = 2000,
        checkpoint_dir: str = "/kaggle/working/checkpoints",
    ):
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir

        # Device & model
        self.model, self.device = set_up_device(model)

        # Optimizer
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
        self._last_loss = None
        self._step_start_time = time.time()
        self.epoch_complete = False

        # Cumulative stats (persisted across runs via checkpoint)
        self._initial_loss = None
        self._cumulative_time = 0.0

        os.makedirs(checkpoint_dir, exist_ok=True)

    @property
    def step(self) -> int:
        return self._step

    @property
    def initial_loss(self) -> float | None:
        return self._initial_loss

    @property
    def cumulative_time(self) -> float:
        return self._cumulative_time

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _status_line(self, loss: float | None = None) -> str:
        """Format a one-line training status summary."""
        lr = self.optimizer.param_groups[0]["lr"]
        parts = [f"step {self._step}/{self.max_steps}"]
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
        parts.append(f"lr {lr:.2e}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def save_checkpoint(self, suffix: str | None = None):
        """
        Save a full training checkpoint (model, optimizer, scaler, step,
        RNG states) and print the current training status.

        Also saves a lightweight model-only snapshot named by step or suffix.
        A 'resume.pt' is always written so that load_checkpoint() can find it.
        """
        name = suffix or f"step_{self._step}"

        model_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        inner_model = unwrap_model(self.model)
        torch.save(inner_model.state_dict(), model_path)
        print(f"Checkpoint saved: {name}")

        # Full resume checkpoint
        resume_state = {
            "step": self._step,
            "last_loss": self._last_loss,
            "status_line": self._status_line(self._last_loss),
            "model_state_dict": inner_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.random.get_rng_state(),
            "initial_loss": self._initial_loss,
            "cumulative_time": self._cumulative_time,
        }

        if self.scaler is not None:
            resume_state["scaler_state_dict"] = self.scaler.state_dict()

        if torch.cuda.is_available():
            resume_state["rng_cuda"] = [
                torch.cuda.get_rng_state(i)
                for i in range(torch.cuda.device_count())
            ]

        resume_path = os.path.join(self.checkpoint_dir, self.RESUME_FILENAME)
        torch.save(resume_state, resume_path)

    def load_checkpoint(self) -> bool:
        """
        Restore training state from a previous checkpoint if one exists.

        Reprints the status line from the checkpoint so the user can see
        where training left off.  Returns True if a checkpoint was loaded,
        False otherwise.
        """
        resume_path = os.path.join(self.checkpoint_dir, self.RESUME_FILENAME)
        if not os.path.isfile(resume_path):
            print("No checkpoint found -- starting from scratch.")
            return False

        ckpt = torch.load(resume_path, map_location=self.device,
                          weights_only=False)

        # Model
        inner_model = unwrap_model(self.model)
        inner_model.load_state_dict(ckpt["model_state_dict"])

        # Optimizer
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Scaler
        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        # Step & loss
        self._step = ckpt["step"]
        # max_steps is per-run: extend ceiling so we train max_steps MORE
        self.max_steps += self._step
        self._last_loss = ckpt.get("last_loss")

        # Cumulative stats
        self._initial_loss = ckpt.get("initial_loss")
        self._cumulative_time = ckpt.get("cumulative_time", 0.0)

        # RNG states
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.random.set_rng_state(ckpt["rng_torch"].cpu())

        if torch.cuda.is_available() and "rng_cuda" in ckpt:
            for i, state in enumerate(ckpt["rng_cuda"]):
                torch.cuda.set_rng_state(state.cpu(), i)

        # Reset timer and log current state with updated max_steps ceiling
        self._step_start_time = time.time()
        self.log()
        return True

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def training_steps(self, data_loader: DataLoader):
        """
        Yield (step, batch) pairs from the data loader up to max_steps.

        Stops when either:
        - max_steps is reached, or
        - the data loader is exhausted (epoch complete for finite datasets)

        Sets self.epoch_complete = True if data was exhausted before max_steps.
        """
        self.epoch_complete = False
        data_iter = iter(data_loader)

        while self._step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch_complete = True
                return
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

        Updates the learning rate, clips gradients, and steps the optimizer.
        Returns True if an optimizer step was taken, False otherwise.
        """
        if self._micro_step < self.grad_accum_steps:
            return False

        # Update LR
        lr = cosine_lr(self._step, self.max_steps, self.learning_rate,
                       self.warmup_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Clip & step
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.optimizer.zero_grad()
        self._last_loss = self._accumulated_loss / self.grad_accum_steps
        self._step += 1
        self._micro_step = 0
        self._accumulated_loss = 0.0
        self._step_start_time = time.time()
        return True

    def should_log(self) -> bool:
        return self._step > 0 and self._step % self.log_interval == 0

    def should_save(self) -> bool:
        return self._step > 0 and self._step % self.save_interval == 0

    def log(self, loss: float = None):
        """Print training metrics for the current step."""
        if loss is None:
            loss = self._last_loss
        elapsed = time.time() - self._step_start_time
        print(f"{self._status_line(loss)} | time {elapsed:.1f}s")

    def record_initial_loss(self, loss: float):
        """Record the very first training loss (only sets once)."""
        if self._initial_loss is None:
            self._initial_loss = loss

    def add_run_time(self, run_time: float):
        """Add this run's elapsed time to the cumulative total."""
        self._cumulative_time += run_time
