"""Write SOURCE_MANIFEST.json: SHA-256 hashes of every tracked source file
(src/configs/tests/specs), for release integrity checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_MANIFEST.json"
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".llm-venv",
    ".hf",
    ".prefetch-venv",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    "results",
    "runlogs",
    "external",
    "frozen",
}
EXCLUDED_SUFFIXES = {".log", ".bundle", ".tmp"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


files = sorted(
    path
    for path in ROOT.rglob("*")
    if path.is_file()
    and path != OUTPUT
    and path.suffix not in EXCLUDED_SUFFIXES
    and not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.parts)
)
manifest = {
    "algorithm": "sha256",
    "files": {
        str(path.relative_to(ROOT)): {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in files
    },
}
OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(OUTPUT)

