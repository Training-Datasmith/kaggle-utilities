"""Tests for training module."""

from contextlib import nullcontext

import torch
import torch.nn as nn
import pytest

from kaggle_utilities.training import (
    LossUnsqueezeWrapper,
    set_up_device,
    cosine_lr,
    get_amp_context,
    get_grad_scaler,
    unwrap_model,
    reduce_loss,
)


class DummyModel(nn.Module):
    """Simple model that returns a scalar loss."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.custom_attr = "test_value"

    def forward(self, **kwargs):
        loss = torch.tensor(1.5)
        return {"loss": loss, "logits": torch.randn(2, 4)}


class TestLossUnsqueezeWrapper:
    def test_unsqueezes_loss(self):
        model = DummyModel()
        wrapped = LossUnsqueezeWrapper(model)
        output = wrapped()
        assert output["loss"].shape == (1,)
        assert output["loss"].item() == pytest.approx(1.5)

    def test_preserves_other_keys(self):
        model = DummyModel()
        wrapped = LossUnsqueezeWrapper(model)
        output = wrapped()
        assert output["logits"].shape == (2, 4)

    def test_handles_none_loss(self):
        class NullLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, **kwargs):
                return {"loss": None, "logits": torch.randn(2, 4)}

        model = NullLossModel()
        wrapped = LossUnsqueezeWrapper(model)
        output = wrapped()
        assert output["loss"] is None

    def test_delegates_attributes(self):
        model = DummyModel()
        wrapped = LossUnsqueezeWrapper(model)
        assert wrapped.custom_attr == "test_value"


class TestSetupDevice:
    def test_cpu_no_wrapping(self):
        model = DummyModel()
        result_model, device = set_up_device(model, use_data_parallel=True)
        if not torch.cuda.is_available():
            assert device == torch.device("cpu")
            assert not isinstance(result_model, nn.DataParallel)
            assert not isinstance(result_model, LossUnsqueezeWrapper)


class TestCosineLR:
    def test_zero_at_start(self):
        lr = cosine_lr(step=0, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        assert lr == 0.0

    def test_warmup_midpoint(self):
        lr = cosine_lr(step=50, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        assert lr == pytest.approx(5e-4)

    def test_max_lr_at_warmup_end(self):
        lr = cosine_lr(step=100, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        assert lr == pytest.approx(1e-3)

    def test_min_lr_at_end(self):
        lr = cosine_lr(
            step=1000, warmup_steps=100, max_steps=1000,
            max_lr=1e-3, min_lr=1e-5,
        )
        assert lr == pytest.approx(1e-5)

    def test_mid_training(self):
        lr = cosine_lr(
            step=550, warmup_steps=100, max_steps=1000,
            max_lr=1e-3, min_lr=1e-5,
        )
        # Should be approximately halfway between max and min
        assert 1e-5 < lr < 1e-3

    def test_beyond_max_steps(self):
        lr = cosine_lr(
            step=2000, warmup_steps=100, max_steps=1000,
            max_lr=1e-3, min_lr=1e-5,
        )
        assert lr == pytest.approx(1e-5)


class TestReduceLoss:
    def test_scalar_passthrough(self):
        loss = torch.tensor(2.5)
        assert reduce_loss(loss).item() == pytest.approx(2.5)

    def test_1d_mean(self):
        loss = torch.tensor([2.0, 4.0])
        assert reduce_loss(loss).item() == pytest.approx(3.0)

    def test_1d_single_element(self):
        loss = torch.tensor([2.5])
        assert reduce_loss(loss).item() == pytest.approx(2.5)


class TestUnwrapModel:
    def test_plain_model(self):
        model = DummyModel()
        assert unwrap_model(model) is model

    def test_loss_wrapper(self):
        model = DummyModel()
        wrapped = LossUnsqueezeWrapper(model)
        assert unwrap_model(wrapped) is model

    def test_double_wrapped(self):
        """Simulate DataParallel(LossUnsqueezeWrapper(model))."""
        model = DummyModel()
        loss_wrapped = LossUnsqueezeWrapper(model)
        # Simulate DataParallel by wrapping with a module attribute
        dp = nn.DataParallel(loss_wrapped)
        assert unwrap_model(dp) is model


class TestGetAmpContext:
    def test_cpu_returns_nullcontext(self):
        ctx = get_amp_context(torch.device("cpu"))
        assert isinstance(ctx, nullcontext)


class TestGetGradScaler:
    def test_cpu_returns_none(self):
        scaler = get_grad_scaler(torch.device("cpu"))
        assert scaler is None
