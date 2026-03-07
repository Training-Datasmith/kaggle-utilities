"""Reusable Python utilities for Kaggle notebooks."""

__version__ = "0.1.0"

from .repo_cloner import clone_repos, collect_source_files
from .composer import resolve_composer_deps, resolve_all_composer_deps
from .dataset import GitHubCodeDataset, build_data_loader
from .training import set_up_device, cosine_lr
from .context import TrainingContext
from .model_store import (
    download_checkpoint,
    ensure_instance,
    ensure_model,
    ensure_version,
    upload_checkpoint,
)
