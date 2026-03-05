"""Git repository cloning and source file collection utilities."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

TRAINING_DATASMITH_REPOS = [
    "magento2", "PrestaShop", "woocommerce", "wordpress-develop", "opencart",
    "zencart", "v6", "thirtybees", "shopware", "oxideshop_ce",
    "oscommerce2", "osCommerce-V4", "PhoenixCart", "oscommerce",
    "bagisto", "aimeos-laravel", "Sylius",
    "symfony", "framework", "cakephp", "drupal", "CodeIgniter4",
    "yii2", "Slim", "laminas-mvc", "fuel", "cphalcon",
]

CODE_EXTENSIONS = {
    ".php", ".inc", ".js", ".ts", ".jsx", ".py",
    ".c", ".h", ".cpp", ".java", ".go", ".rs",
    ".twig", ".blade.php", ".json", ".xml", ".yaml", ".yml",
}

SKIP_DIRS = {
    "vendor", "node_modules", ".git", "test", "tests", "Test", "Tests",
    "__pycache__", "fixtures", "cache", "logs",
}


def clone_repos(
    repos: list[str] | None = None,
    org: str = "Training-Datasmith",
    dest_dir: str = "/kaggle/working/repos",
    depth: int = 1,
    branch: str | None = None,
) -> dict[str, bool]:
    """
    Shallow-clone repositories from a GitHub organization.

    Args:
        repos: List of repo names. Defaults to TRAINING_DATASMITH_REPOS.
        org: GitHub organization name.
        dest_dir: Directory to clone into.
        depth: Git clone depth (1 = latest snapshot only).
        branch: Specific branch to clone. None = default branch.

    Returns:
        Dict mapping repo name to success boolean.
    """
    if repos is None:
        repos = TRAINING_DATASMITH_REPOS

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}
    for repo in repos:
        repo_path = dest / repo
        if repo_path.exists():
            print(f"  [skip] {repo} already exists")
            results[repo] = True
            continue

        url = f"https://github.com/{org}/{repo}.git"
        cmd = ["git", "clone", "--depth", str(depth)]
        if branch:
            cmd += ["--branch", branch]
        cmd += [url, str(repo_path)]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            print(f"  [ok] {repo}")
            results[repo] = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  [FAIL] {repo}: {e}")
            results[repo] = False

    return results


def collect_source_files(
    repo_dir: str = "/kaggle/working/repos",
    extensions: set[str] | None = None,
    skip_dirs: set[str] | None = None,
    min_size: int = 100,
    max_size: int = 500_000,
    shuffle: bool = True,
) -> list[str]:
    """
    Walk cloned repos and collect source file paths.

    Args:
        repo_dir: Root directory containing cloned repos.
        extensions: Set of file extensions to include. Defaults to CODE_EXTENSIONS.
        skip_dirs: Set of directory names to skip. Defaults to SKIP_DIRS.
        min_size: Minimum file size in bytes.
        max_size: Maximum file size in bytes.
        shuffle: Whether to shuffle the file list.

    Returns:
        List of absolute file paths.
    """
    if extensions is None:
        extensions = CODE_EXTENSIONS
    if skip_dirs is None:
        skip_dirs = SKIP_DIRS

    files: list[str] = []
    root = Path(repo_dir)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            # Check extension (handle compound like .blade.php)
            if not any(fname.endswith(ext) for ext in extensions):
                continue

            try:
                size = fpath.stat().st_size
            except OSError:
                continue

            if min_size <= size <= max_size:
                files.append(str(fpath.resolve()))

    if shuffle:
        random.shuffle(files)

    return files
