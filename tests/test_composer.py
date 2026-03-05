"""Tests for composer module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kaggle_utilities.composer import (
    KNOWN_PACKAGE_MAP,
    parse_composer_json,
    map_package_to_repo,
    resolve_composer_deps,
    resolve_all_composer_deps,
)


SAMPLE_COMPOSER = {
    "require": {
        "php": ">=8.2",
        "ext-json": "*",
        "fuelphp/foundation": "^2.0",
        "psr/log": "^3.0",
        "monolog/monolog": "^3.0",
    },
    "require-dev": {
        "ext-mbstring": "*",
        "phpunit/phpunit": "^10.0",
    },
}


class TestParseComposerJson:
    def test_parses_require_and_require_dev(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(SAMPLE_COMPOSER, f)
            f.flush()
            packages = parse_composer_json(f.name)

        os.unlink(f.name)

        names = [p["name"] for p in packages]
        assert "fuelphp/foundation" in names
        assert "psr/log" in names
        assert "monolog/monolog" in names
        assert "phpunit/phpunit" in names

    def test_filters_php_and_extensions(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(SAMPLE_COMPOSER, f)
            f.flush()
            packages = parse_composer_json(f.name)

        os.unlink(f.name)

        names = [p["name"] for p in packages]
        assert "php" not in names
        assert "ext-json" not in names
        assert "ext-mbstring" not in names

    def test_returns_version_constraints(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(SAMPLE_COMPOSER, f)
            f.flush()
            packages = parse_composer_json(f.name)

        os.unlink(f.name)

        psr = next(p for p in packages if p["name"] == "psr/log")
        assert psr["version_constraint"] == "^3.0"

    def test_missing_file_returns_empty(self):
        packages = parse_composer_json("/nonexistent/composer.json")
        assert packages == []

    def test_no_require_section(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump({"name": "test/pkg"}, f)
            f.flush()
            packages = parse_composer_json(f.name)

        os.unlink(f.name)
        assert packages == []


class TestMapPackageToRepo:
    @patch("kaggle_utilities.composer._repo_exists", return_value=False)
    def test_known_map_match(self, mock_exists):
        result = map_package_to_repo("fuelphp/fuel", "Training-Datasmith")
        assert result == "fuel"
        mock_exists.assert_not_called()

    @patch("kaggle_utilities.composer._repo_exists", return_value=False)
    def test_known_map_laravel(self, mock_exists):
        result = map_package_to_repo("laravel/framework", "Training-Datasmith")
        assert result == "framework"

    @patch("kaggle_utilities.composer._repo_exists")
    def test_exact_match_on_package_name(self, mock_exists):
        mock_exists.side_effect = lambda org, repo: repo == "foundation"
        result = map_package_to_repo("fuelphp/foundation", "Training-Datasmith")
        assert result == "foundation"

    @patch("kaggle_utilities.composer._repo_exists")
    def test_vendor_prefixed_match(self, mock_exists):
        # First call (exact match) returns False, second (vendor-prefix) returns True
        mock_exists.side_effect = lambda org, repo: repo == "fuelphp-core"
        result = map_package_to_repo("fuelphp/core", "Training-Datasmith")
        assert result == "fuelphp-core"

    @patch("kaggle_utilities.composer._repo_exists", return_value=False)
    def test_unknown_package_returns_none(self, mock_exists):
        result = map_package_to_repo("unknown/package", "Training-Datasmith")
        assert result is None

    @patch("kaggle_utilities.composer._repo_exists", return_value=False)
    def test_custom_known_map(self, mock_exists):
        custom = {"custom/pkg": "my-repo"}
        result = map_package_to_repo(
            "custom/pkg", "Org", known_map=custom,
        )
        assert result == "my-repo"

    @patch("kaggle_utilities.composer._repo_exists", return_value=False)
    def test_invalid_package_name(self, mock_exists):
        result = map_package_to_repo("no-slash", "Org")
        assert result is None


class TestResolveComposerDeps:
    @patch("kaggle_utilities.composer._repo_exists", return_value=False)
    def test_no_composer_json(self, mock_exists):
        with tempfile.TemporaryDirectory() as tmpdir:
            cloned, resolved = resolve_composer_deps(
                tmpdir, clone_dir=tmpdir,
            )
            assert cloned == []

    @patch("kaggle_utilities.composer.subprocess.run")
    @patch("kaggle_utilities.composer._repo_exists")
    def test_already_resolved_skipped(self, mock_exists, mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create composer.json
            composer = {"require": {"psr/log": "^3.0"}}
            Path(tmpdir, "composer.json").write_text(json.dumps(composer))

            already = {"psr/log"}
            cloned, resolved = resolve_composer_deps(
                tmpdir,
                clone_dir=tmpdir,
                already_resolved=already,
            )
            assert cloned == []
            mock_run.assert_not_called()

    @patch("kaggle_utilities.composer.subprocess.run")
    @patch("kaggle_utilities.composer._repo_exists", return_value=True)
    def test_clones_found_dependency(self, mock_exists, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "myrepo"
            repo_dir.mkdir()
            composer = {"require": {"vendor/deplib": "^1.0"}}
            (repo_dir / "composer.json").write_text(json.dumps(composer))

            clone_dir = Path(tmpdir) / "clones"
            clone_dir.mkdir()

            cloned, resolved = resolve_composer_deps(
                str(repo_dir),
                clone_dir=str(clone_dir),
                recursive=False,
            )
            assert len(cloned) == 1
            assert "vendor/deplib" in resolved


class TestResolveAllComposerDeps:
    @patch("kaggle_utilities.composer.subprocess.run")
    @patch("kaggle_utilities.composer._repo_exists", return_value=True)
    def test_deduplicates_shared_deps(self, mock_exists, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two repos that share a dependency
            for name in ("repo1", "repo2"):
                repo_dir = Path(tmpdir) / name
                repo_dir.mkdir()
                composer = {"require": {"psr/log": "^3.0"}}
                (repo_dir / "composer.json").write_text(json.dumps(composer))

            clone_dir = Path(tmpdir) / "clones"
            clone_dir.mkdir()

            repo_dirs = [
                str(Path(tmpdir) / "repo1"),
                str(Path(tmpdir) / "repo2"),
            ]
            cloned, skipped = resolve_all_composer_deps(
                repo_dirs,
                clone_dir=str(clone_dir),
                recursive=False,
            )
            # psr/log should appear in skipped_duplicates (requested 2x)
            assert "psr/log" in skipped
            # But only cloned once
            assert mock_run.call_count == 1
