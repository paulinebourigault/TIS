"""Write SOURCE_MANIFEST.json: SHA-256 hashes of every git-tracked source file
(results and frozen artifacts excluded), for release integrity checks."""
from __future__ import annotations

import hashlib
import json
import subprocess
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


tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                         capture_output=True).stdout.decode().split("\0")
files = sorted(
    ROOT / name
    for name in tracked
    if name
    and (ROOT / name).is_file()
    and (ROOT / name) != OUTPUT
    and Path(name).suffix not in EXCLUDED_SUFFIXES
    and not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in Path(name).parts)
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

