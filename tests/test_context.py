"""Tests for TrainingContext."""

import os
import math

import torch
import torch.nn as nn
import pytest

from kaggle_utilities.context import TrainingContext
from kaggle_utilities.training import inverse_sqrt_lr


class TinyModel(nn.Module):
    """Minimal model for testing TrainingContext."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, input_ids, labels=None):
        x = self.linear(input_ids.float())
        loss = x.sum() if labels is not None else None
        return {"logits": x, "loss": loss}


class TestContextUsesInverseSqrtLR:
    def test_lr_matches_inverse_sqrt_schedule(self, tmp_path):
        model = TinyModel()
        ctx = TrainingContext(
            model=model,
            max_steps=100,
            learning_rate=3e-4,
            warmup_steps=10,
            grad_accum_steps=1,
            checkpoint_dir=str(tmp_path),
        )

        loader = [{"input_ids": torch.randn(1, 8), "labels": torch.ones(1, 8)}] * 50
        for step, batch in ctx.training_steps(loader):
            ctx.forward_backward(batch)
            if ctx.step_optimizer():
                actual_lr = ctx.optimizer.param_groups[0]["lr"]
                expected_lr = inverse_sqrt_lr(ctx.step - 1, 3e-4, 10)
                assert actual_lr == pytest.approx(expected_lr), (
                    f"step {ctx.step}: lr {actual_lr} != expected {expected_lr}"
                )

    def test_lr_continues_after_resume(self, tmp_path):
        """LR should not reset or jump when resuming from checkpoint."""
        model = TinyModel()
        ctx = TrainingContext(
            model=model,
            max_steps=20,
            learning_rate=3e-4,
            warmup_steps=5,
            grad_accum_steps=1,
            checkpoint_dir=str(tmp_path),
        )

        loader = [{"input_ids": torch.randn(1, 8), "labels": torch.ones(1, 8)}] * 50
        for step, batch in ctx.training_steps(loader):
            ctx.forward_backward(batch)
            ctx.step_optimizer()
        ctx.save_checkpoint()

        lr_before_save = ctx.optimizer.param_groups[0]["lr"]
        step_before_save = ctx.step

        # Resume in a new context
        model2 = TinyModel()
        ctx2 = TrainingContext(
            model=model2,
            max_steps=20,
            learning_rate=3e-4,
            warmup_steps=5,
            grad_accum_steps=1,
            checkpoint_dir=str(tmp_path),
        )
        ctx2.load_checkpoint()

        assert ctx2.step == step_before_save

        # Take one more step and verify LR is continuous
        for step, batch in ctx2.training_steps(loader):
            ctx2.forward_backward(batch)
            if ctx2.step_optimizer():
                lr_after_resume = ctx2.optimizer.param_groups[0]["lr"]
                expected = inverse_sqrt_lr(ctx2.step - 1, 3e-4, 5)
                assert lr_after_resume == pytest.approx(expected)
                break

    def test_no_step_estimate_parameter(self):
        """TrainingContext should not accept a step_estimate parameter."""
        model = TinyModel()
        with pytest.raises(TypeError, match="step_estimate"):
            TrainingContext(model=model, step_estimate=60000)
