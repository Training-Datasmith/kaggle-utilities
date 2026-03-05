"""Tests for repo_cloner module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kaggle_utilities.repo_cloner import (
    TRAINING_DATASMITH_REPOS,
    CODE_EXTENSIONS,
    SKIP_DIRS,
    clone_repos,
    collect_source_files,
)


class TestConstants:
    def test_repos_list_not_empty(self):
        assert len(TRAINING_DATASMITH_REPOS) > 0

    def test_repos_are_strings(self):
        for repo in TRAINING_DATASMITH_REPOS:
            assert isinstance(repo, str)
            assert len(repo) > 0

    def test_code_extensions_not_empty(self):
        assert len(CODE_EXTENSIONS) > 0

    def test_code_extensions_start_with_dot(self):
        for ext in CODE_EXTENSIONS:
            assert ext.startswith(".")

    def test_skip_dirs_not_empty(self):
        assert len(SKIP_DIRS) > 0

    def test_skip_dirs_includes_common(self):
        assert "vendor" in SKIP_DIRS
        assert "node_modules" in SKIP_DIRS
        assert ".git" in SKIP_DIRS


class TestCloneRepos:
    @patch("kaggle_utilities.repo_cloner.subprocess.run")
    def test_clone_repos_calls_git(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = clone_repos(
                repos=["test-repo"],
                org="TestOrg",
                dest_dir=tmpdir,
            )
            assert results["test-repo"] is True
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "git" in cmd
            assert "clone" in cmd
            assert "https://github.com/TestOrg/test-repo.git" in cmd

    @patch("kaggle_utilities.repo_cloner.subprocess.run")
    def test_clone_repos_with_branch(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            clone_repos(
                repos=["repo1"],
                org="Org",
                dest_dir=tmpdir,
                branch="main",
            )
            cmd = mock_run.call_args[0][0]
            assert "--branch" in cmd
            assert "main" in cmd

    @patch("kaggle_utilities.repo_cloner.subprocess.run")
    def test_clone_repos_skips_existing(self, mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create the repo dir
            os.makedirs(os.path.join(tmpdir, "existing-repo"))
            results = clone_repos(
                repos=["existing-repo"],
                dest_dir=tmpdir,
            )
            assert results["existing-repo"] is True
            mock_run.assert_not_called()

    @patch("kaggle_utilities.repo_cloner.subprocess.run")
    def test_clone_repos_handles_failure(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        with tempfile.TemporaryDirectory() as tmpdir:
            results = clone_repos(
                repos=["bad-repo"],
                dest_dir=tmpdir,
            )
            assert results["bad-repo"] is False

    @patch("kaggle_utilities.repo_cloner.subprocess.run")
    def test_clone_repos_defaults_to_training_datasmith(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = clone_repos(dest_dir=tmpdir)
            assert len(results) == len(TRAINING_DATASMITH_REPOS)


class TestCollectSourceFiles:
    def _create_test_files(self, root: Path):
        """Create a directory tree with various test files."""
        # Valid PHP file
        php_dir = root / "repo1" / "src"
        php_dir.mkdir(parents=True)
        (php_dir / "main.php").write_text("<?php echo 'hello'; ?>" * 10)

        # Valid JS file
        (php_dir / "app.js").write_text("console.log('hello');" * 10)

        # Too small file
        (php_dir / "tiny.php").write_text("<?php")

        # File in vendor (should be skipped)
        vendor_dir = root / "repo1" / "vendor" / "pkg"
        vendor_dir.mkdir(parents=True)
        (vendor_dir / "lib.php").write_text("<?php vendor code;" * 10)

        # Non-code file
        (php_dir / "image.png").write_bytes(b"\x89PNG" + b"\x00" * 200)

        # File in .git (should be skipped)
        git_dir = root / "repo1" / ".git"
        git_dir.mkdir(parents=True)
        (git_dir / "config").write_text("git config" * 20)

    def test_collects_php_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_test_files(root)
            files = collect_source_files(tmpdir, shuffle=False)
            extensions = {Path(f).suffix for f in files}
            assert ".php" in extensions or ".js" in extensions

    def test_filters_by_min_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_test_files(root)
            files = collect_source_files(tmpdir, min_size=100, shuffle=False)
            for f in files:
                assert os.path.getsize(f) >= 100

    def test_skips_vendor_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_test_files(root)
            files = collect_source_files(tmpdir, shuffle=False)
            for f in files:
                assert "vendor" not in Path(f).parts

    def test_skips_git_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_test_files(root)
            files = collect_source_files(tmpdir, shuffle=False)
            for f in files:
                assert ".git" not in Path(f).parts

    def test_skips_non_code_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_test_files(root)
            files = collect_source_files(tmpdir, shuffle=False)
            for f in files:
                assert not f.endswith(".png")

    def test_custom_extensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_test_files(root)
            files = collect_source_files(
                tmpdir, extensions={".js"}, shuffle=False,
            )
            for f in files:
                assert f.endswith(".js")

    def test_shuffle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create enough files for shuffle to be meaningful
            src = root / "repo"
            src.mkdir(parents=True)
            for i in range(20):
                (src / f"file{i:02d}.php").write_text(f"<?php // file {i}" * 20)

            files1 = collect_source_files(tmpdir, shuffle=True)
            files2 = collect_source_files(tmpdir, shuffle=False)
            # Both should have same files
            assert sorted(files1) == sorted(files2)
