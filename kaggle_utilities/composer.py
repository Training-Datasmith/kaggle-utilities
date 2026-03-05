"""PHP Composer dependency resolution for GitHub organizations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Curated mapping for packages whose Composer names don't map cleanly
# to repo names. Covers the most common frameworks in our org.
KNOWN_PACKAGE_MAP = {
    "fuelphp/fuel": "fuel",
    "laravel/framework": "framework",
    "symfony/symfony": "symfony",
}


def parse_composer_json(composer_json_path: str) -> list[dict]:
    """
    Parse composer.json and extract require/require-dev package names.

    Filters out PHP extensions (ext-*) and php version constraints.

    Returns:
        List of dicts with keys: 'name' (vendor/package), 'version_constraint'.
    """
    path = Path(composer_json_path)
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    packages: list[dict] = []
    for section in ("require", "require-dev"):
        deps = data.get(section, {})
        for name, version in deps.items():
            # Skip php itself and extensions
            if name == "php" or name.startswith("ext-"):
                continue
            packages.append({
                "name": name,
                "version_constraint": version,
            })

    return packages


def map_package_to_repo(
    package_name: str,
    org: str,
    known_map: dict[str, str] | None = None,
) -> str | None:
    """
    Attempt to map a Composer package name (e.g., 'fuelphp/foundation')
    to a GitHub repo name in the given org.

    Tries several strategies in order:
    1. Known mapping table (KNOWN_PACKAGE_MAP or custom override)
    2. Exact match on package part (foundation -> foundation)
    3. Vendor match (fuelphp/* -> check org repos starting with fuelphp-)

    Args:
        package_name: Composer package name (vendor/package format).
        org: GitHub organization name.
        known_map: Optional override for the known mapping table.
            Defaults to KNOWN_PACKAGE_MAP.

    Returns:
        Repo name if found, None otherwise.
    """
    if known_map is None:
        known_map = KNOWN_PACKAGE_MAP

    # Strategy 1: Known mapping
    if package_name in known_map:
        return known_map[package_name]

    # Parse vendor/package
    parts = package_name.split("/", 1)
    if len(parts) != 2:
        return None
    vendor, package = parts

    # Strategy 2: Check if repo with the package name exists in the org
    if _repo_exists(org, package):
        return package

    # Strategy 3: Check vendor-prefixed repo name
    vendor_prefixed = f"{vendor}-{package}"
    if _repo_exists(org, vendor_prefixed):
        return vendor_prefixed

    return None


def _repo_exists(org: str, repo: str) -> bool:
    """Check if a GitHub repo exists using gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "repo", "view", f"{org}/{repo}", "--json", "name"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def resolve_composer_deps(
    repo_path: str,
    org: str = "Training-Datasmith",
    clone_dir: str = "/kaggle/working/repos",
    depth: int = 1,
    recursive: bool = True,
    already_resolved: set[str] | None = None,
) -> tuple[list[str], set[str]]:
    """
    Read composer.json from a single repo and clone any dependency
    repos that exist in the given GitHub org.

    Args:
        repo_path: Path to the cloned repo with a composer.json.
        org: GitHub org to check for dependency repos.
        clone_dir: Where to clone discovered dependencies.
        depth: Git clone depth.
        recursive: If True, also resolve dependencies of dependencies.
        already_resolved: Set of package names already cloned (for
            deduplication across multiple repos). Updated in place.

    Returns:
        (newly_cloned_paths, already_resolved) -- the set is passed
        through so callers can chain calls and maintain dedup state.
    """
    if already_resolved is None:
        already_resolved = set()

    composer_json = Path(repo_path) / "composer.json"
    packages = parse_composer_json(str(composer_json))

    newly_cloned: list[str] = []
    clone_root = Path(clone_dir)
    clone_root.mkdir(parents=True, exist_ok=True)

    for pkg in packages:
        pkg_name = pkg["name"]

        if pkg_name in already_resolved:
            continue

        repo_name = map_package_to_repo(pkg_name, org)
        if repo_name is None:
            already_resolved.add(pkg_name)
            continue

        dest = clone_root / repo_name
        if dest.exists():
            already_resolved.add(pkg_name)
            continue

        # Clone the dependency
        url = f"https://github.com/{org}/{repo_name}.git"
        cmd = ["git", "clone", "--depth", str(depth), url, str(dest)]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            print(f"  [dep] {pkg_name} -> {repo_name}")
            already_resolved.add(pkg_name)
            newly_cloned.append(str(dest))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  [dep-fail] {pkg_name}: {e}")
            already_resolved.add(pkg_name)
            continue

    # Recursively resolve deps of newly cloned repos
    if recursive:
        for cloned_path in list(newly_cloned):
            sub_cloned, already_resolved = resolve_composer_deps(
                cloned_path,
                org=org,
                clone_dir=clone_dir,
                depth=depth,
                recursive=True,
                already_resolved=already_resolved,
            )
            newly_cloned.extend(sub_cloned)

    return newly_cloned, already_resolved


def resolve_all_composer_deps(
    repo_dirs: list[str],
    org: str = "Training-Datasmith",
    clone_dir: str = "/kaggle/working/repos",
    depth: int = 1,
    recursive: bool = True,
) -> tuple[list[str], set[str]]:
    """
    Resolve Composer dependencies across ALL repos, deduplicating.

    Multiple repos may depend on the same package (e.g., psr/log).
    This function ensures each dependency is cloned at most once,
    preventing duplicate training data that would bias the model
    toward commonly-shared packages.

    Args:
        repo_dirs: List of paths to cloned repos to scan.
        org: GitHub org to check for dependency repos.
        clone_dir: Where to clone discovered dependencies.
        depth: Git clone depth.
        recursive: If True, also resolve deps of newly cloned deps.

    Returns:
        (newly_cloned_paths, skipped_duplicates) -- the list of new
        repo paths that were cloned, and the set of package names
        that were requested by multiple repos but only cloned once.
    """
    already_resolved: set[str] = set()
    all_cloned: list[str] = []
    requested_counts: dict[str, int] = {}

    for repo_dir in repo_dirs:
        composer_json = Path(repo_dir) / "composer.json"
        if not composer_json.exists():
            continue

        # Track how many repos request each package
        packages = parse_composer_json(str(composer_json))
        for pkg in packages:
            requested_counts[pkg["name"]] = requested_counts.get(pkg["name"], 0) + 1

        cloned, already_resolved = resolve_composer_deps(
            repo_dir,
            org=org,
            clone_dir=clone_dir,
            depth=depth,
            recursive=recursive,
            already_resolved=already_resolved,
        )
        all_cloned.extend(cloned)

    # Packages requested by 2+ repos (deduplicated by resolve_composer_deps)
    skipped_duplicates = {
        name for name, count in requested_counts.items() if count > 1
    }

    return all_cloned, skipped_duplicates
