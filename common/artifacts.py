"""File digests, recorded in every result JSON and printed by every runner.

Each run stores the sha256 of the model files and of the input clip it actually used,
so a result can always be traced back to the exact artefacts that produced it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()