# kaggle-utilities

Reusable Python utilities for Kaggle notebooks in the
[Training-Datasmith](https://github.com/Training-Datasmith) organization.

## Installation

On Kaggle, install directly from GitHub:

```bash
pip install git+https://github.com/Training-Datasmith/kaggle-utilities.git
```

For local development:

```bash
git clone https://github.com/Training-Datasmith/kaggle-utilities.git
cd kaggle-utilities
pip install -e ".[dev]"
```

## Modules

- **`repo_cloner`** — Clone repos from a GitHub org and collect source files
- **`composer`** — Resolve PHP Composer dependencies to find additional repos
- **`dataset`** — `GitHubCodeDataset` for streaming tokenized code into training batches
- **`training`** — Device setup (single/multi-GPU), AMP, LR schedule, gradient accumulation
- **`model`** — OLMo3Mini reference implementation with GQA, SwiGLU, RoPE

## Quick Start

```python
from kaggle_utilities import clone_repos, collect_source_files, build_data_loader
from kaggle_utilities.training import set_up_device, TrainingContext

# Clone repos
clone_repos()
files = collect_source_files()

# Build data pipeline
loader, tokenizer = build_data_loader(files)

# Set up training
model = YourModel()
model, device = set_up_device(model)
ctx = TrainingContext(model=model, max_steps=2000)
```

## Running Tests

```bash
pytest tests/ -v
```

## License

LGPL-2.1. See [LICENSE](LICENSE).
