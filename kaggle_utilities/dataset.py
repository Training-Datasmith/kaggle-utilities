"""Dataset and data pipeline utilities for LM training on source code."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


class GitHubCodeDataset(IterableDataset):
    """
    Tokenize and pack source files into fixed-length sequences for LM training.

    Streams through files, tokenizes on-the-fly, packs tokens into
    (seq_len+1) chunks yielding input_ids and labels for next-token prediction.
    Loops through files indefinitely (infinite dataset).
    """

    def __init__(
        self,
        file_paths: list[str],
        tokenizer,
        seq_len: int = 2048,
        shuffle_files: bool = True,
    ):
        self.file_paths = list(file_paths)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_files = shuffle_files

    def _token_stream(self):
        """Yield tokens from files, looping indefinitely."""
        paths = list(self.file_paths)
        while True:
            if self.shuffle_files:
                random.shuffle(paths)
            for fpath in paths:
                try:
                    text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if not text.strip():
                    continue
                tokens = self.tokenizer.encode(text)
                yield from tokens

    def __iter__(self):
        """Yield dicts with input_ids and labels tensors."""
        buf: list[int] = []
        chunk_len = self.seq_len + 1

        for token in self._token_stream():
            buf.append(token)
            if len(buf) == chunk_len:
                t = torch.tensor(buf, dtype=torch.long)
                yield {
                    "input_ids": t[:-1],
                    "labels": t[1:],
                }
                buf = []


def build_data_loader(
    file_paths: list[str],
    tokenizer_id: str = "allenai/OLMo-2-0425-1B",
    seq_len: int = 2048,
    batch_size: int = 2,
    num_workers: int = 0,
) -> tuple[DataLoader, AutoTokenizer]:
    """
    Convenience function: create tokenizer, dataset, and DataLoader.

    Returns:
        (DataLoader, tokenizer) tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    dataset = GitHubCodeDataset(
        file_paths=file_paths,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return loader, tokenizer
