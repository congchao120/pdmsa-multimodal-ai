from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    try:
        import torch

        worker_seed = torch.initial_seed() % (2**32)
    except ImportError:
        worker_seed = worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a frozen input artifact without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
