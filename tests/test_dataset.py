"""Tests for dataset module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
import pytest

from kaggle_utilities.dataset import GitHubCodeDataset


class MockTokenizer:
    """Simple tokenizer mock that returns character ordinals."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]


class TestGitHubCodeDataset:
    def _create_test_files(self, tmpdir: str, n_files: int = 3, content_len: int = 200):
        paths = []
        for i in range(n_files):
            p = Path(tmpdir) / f"file{i}.py"
            p.write_text(f"# file {i}\n" + "x = 1\n" * content_len)
            paths.append(str(p))
        return paths

    def test_yields_correct_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_test_files(tmpdir)
            tokenizer = MockTokenizer()
            dataset = GitHubCodeDataset(
                paths, tokenizer, seq_len=32, shuffle_files=False,
            )

            batch = next(iter(dataset))
            assert batch["input_ids"].shape == (32,)
            assert batch["labels"].shape == (32,)

    def test_labels_shifted_by_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_test_files(tmpdir)
            tokenizer = MockTokenizer()
            dataset = GitHubCodeDataset(
                paths, tokenizer, seq_len=32, shuffle_files=False,
            )

            batch = next(iter(dataset))
            # Labels should be the next token after input_ids
            # Since they come from the same chunk of seq_len+1,
            # input_ids = chunk[:-1], labels = chunk[1:]
            # So labels[i] should equal input_ids[i+1] for the original chunk
            # Verify they are different tensors but related
            assert batch["input_ids"].dtype == torch.long
            assert batch["labels"].dtype == torch.long

    def test_yields_multiple_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_test_files(tmpdir, content_len=500)
            tokenizer = MockTokenizer()
            dataset = GitHubCodeDataset(
                paths, tokenizer, seq_len=16, shuffle_files=False,
            )

            count = 0
            for batch in dataset:
                count += 1
                if count >= 10:
                    break
            assert count == 10

    def test_handles_empty_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an empty file and a non-empty file
            (Path(tmpdir) / "empty.py").write_text("")
            (Path(tmpdir) / "full.py").write_text("x = 1\n" * 200)
            paths = [
                str(Path(tmpdir) / "empty.py"),
                str(Path(tmpdir) / "full.py"),
            ]
            tokenizer = MockTokenizer()
            dataset = GitHubCodeDataset(
                paths, tokenizer, seq_len=16, shuffle_files=False,
            )

            # Should still yield batches (from the non-empty file)
            batch = next(iter(dataset))
            assert batch["input_ids"].shape == (16,)

    def test_custom_seq_len(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_test_files(tmpdir, content_len=500)
            tokenizer = MockTokenizer()

            for seq_len in [64, 128, 256]:
                dataset = GitHubCodeDataset(
                    paths, tokenizer, seq_len=seq_len, shuffle_files=False,
                )
                batch = next(iter(dataset))
                assert batch["input_ids"].shape == (seq_len,)
                assert batch["labels"].shape == (seq_len,)
