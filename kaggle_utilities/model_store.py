"""Kaggle Models API helpers for checkpoint persistence."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run a kaggle CLI command and return the result."""
    return subprocess.run(args, capture_output=True, text=True)


def _parse_versions(stdout: str) -> list[str]:
    """Extract version numbers from kaggle versions list output.

    Skips header and separator lines (dashes); returns version strings
    in the order returned (latest first).
    """
    return [
        line.split()[0]
        for line in stdout.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]


def model_exists(owner: str, slug: str) -> bool:
    """Check whether a Kaggle model (and at least one instance) exists."""
    result = _run(["kaggle", "models", "instances", "list", f"{owner}/{slug}"])
    return result.returncode == 0


def ensure_model(
    owner: str,
    slug: str,
    title: str,
    *,
    description: str = "",
    private: bool = True,
    upload_dir: str = "/kaggle/working/model-upload",
) -> None:
    """Create a Kaggle model if it doesn't already exist."""
    if model_exists(owner, slug):
        print(f"Model '{owner}/{slug}' already exists.")
        return

    os.makedirs(upload_dir, exist_ok=True)
    meta = {
        "ownerSlug": owner,
        "title": title,
        "slug": slug,
        "subtitle": "",
        "isPrivate": private,
        "description": description,
        "publishTime": "",
        "provenanceSources": "",
    }
    meta_path = os.path.join(upload_dir, "model-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    result = _run(["kaggle", "models", "create", "-p", upload_dir])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create model: {result.stderr}")
    print(f"Created model '{owner}/{slug}'.")


def instance_exists(owner: str, slug: str) -> bool:
    """Check whether a Kaggle model instance exists (has data rows)."""
    result = _run(["kaggle", "models", "instances", "list", f"{owner}/{slug}"])
    if result.returncode != 0:
        return False
    return any(
        line.strip() and line.strip()[0].isdigit()
        for line in result.stdout.strip().split("\n")
    )


def ensure_instance(
    owner: str,
    slug: str,
    *,
    instance_slug: str = "default",
    framework: str = "pytorch",
    overview: str = "Training checkpoint",
    license_name: str = "Apache 2.0",
    upload_dir: str = "/kaggle/working/model-upload",
) -> None:
    """Create a Kaggle model instance if it doesn't already exist."""
    if instance_exists(owner, slug):
        print(f"Instance '{owner}/{slug}/{framework}/{instance_slug}' already exists.")
        return

    os.makedirs(upload_dir, exist_ok=True)
    meta = {
        "ownerSlug": owner,
        "modelSlug": slug,
        "instanceSlug": instance_slug,
        "framework": framework,
        "overview": overview,
        "licenseName": license_name,
        "fineTunable": False,
    }
    meta_path = os.path.join(upload_dir, "model-instance-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    result = _run(["kaggle", "models", "instances", "create", "-p", upload_dir])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create instance: {result.stderr}")
    print(f"Created instance '{owner}/{slug}/{framework}/{instance_slug}'.")


def list_versions(handle: str) -> list[str]:
    """List version numbers for a model handle (latest first).

    handle: owner/model-slug/framework/instance-slug
    """
    result = _run(
        ["kaggle", "models", "instances", "versions", "list", handle]
    )
    if result.returncode != 0:
        return []
    return _parse_versions(result.stdout)


def upload_checkpoint(
    handle: str,
    checkpoint_dir: str,
    *,
    note: str = "",
    create_if_missing: bool = False,
    upload_dir: str = "/kaggle/working/model-upload",
) -> None:
    """Upload checkpoint files as a new model version.

    handle: owner/model-slug/framework/instance-slug
    checkpoint_dir: directory containing resume.pt (and any other files to upload)
    note: version note
    create_if_missing: if True, create model and instance when they don't exist
    upload_dir: staging directory for the upload
    """
    parts = handle.split("/")
    if len(parts) != 4:
        raise ValueError(f"Handle must be owner/model/framework/instance, got: {handle}")
    owner, slug = parts[0], parts[1]

    if create_if_missing:
        title = slug.replace("-", " ").title()
        ensure_model(owner, slug, title, upload_dir=upload_dir)
        ensure_instance(owner, slug, instance_slug=parts[3],
                        framework=parts[2], upload_dir=upload_dir)
    elif not instance_exists(owner, slug):
        raise RuntimeError(
            f"Model instance '{handle}' not found. "
            "Use create_if_missing=True to create it automatically."
        )

    # Stage files
    os.makedirs(upload_dir, exist_ok=True)
    resume_src = os.path.join(checkpoint_dir, "resume.pt")
    resume_dst = os.path.join(upload_dir, "resume.pt")
    if os.path.isfile(resume_src):
        import shutil
        shutil.copy2(resume_src, resume_dst)
    else:
        raise FileNotFoundError(f"No resume.pt found in {checkpoint_dir}")

    args = ["kaggle", "models", "instances", "versions", "create",
            handle, "-p", upload_dir]
    if note:
        args.extend(["-n", note])

    result = _run(args)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to upload: {result.stderr}")
    print(f"Uploaded checkpoint to {handle} ({note or 'no note'})")


def download_checkpoint(
    handle: str,
    dest_dir: str,
) -> bool:
    """Download the latest model version and extract into dest_dir.

    handle: owner/model-slug/framework/instance-slug
    dest_dir: where to extract resume.pt

    Returns True if a checkpoint was downloaded, False if no versions exist.
    """
    versions = list_versions(handle)
    if not versions:
        print(f"No versions found for {handle}.")
        return False

    latest = versions[0]
    versioned_handle = f"{handle}/{latest}"

    os.makedirs(dest_dir, exist_ok=True)
    result = _run([
        "kaggle", "models", "instances", "versions", "download",
        versioned_handle, "-p", dest_dir,
    ])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download: {result.stderr}")

    # Extract tar.gz files
    for fname in os.listdir(dest_dir):
        if fname.endswith(".tar.gz"):
            fpath = os.path.join(dest_dir, fname)
            with tarfile.open(fpath) as tar:
                tar.extractall(dest_dir, filter="data")
            os.remove(fpath)

    print(f"Downloaded checkpoint from {handle} (version {latest})")
    return True


def ensure_version(
    handle: str,
    checkpoint_dir: str,
    *,
    note: str = "",
    create_if_missing: bool = True,
    upload_dir: str = "/kaggle/working/model-upload",
) -> None:
    """Upload a checkpoint version only if none exists yet.

    Useful for testing — ensures the model/instance/version chain exists
    without creating duplicate versions on repeated runs.
    """
    versions = list_versions(handle)
    if versions:
        print(f"Version {versions[0]} already exists for {handle}, skipping upload.")
        return

    upload_checkpoint(
        handle, checkpoint_dir,
        note=note,
        create_if_missing=create_if_missing,
        upload_dir=upload_dir,
    )
