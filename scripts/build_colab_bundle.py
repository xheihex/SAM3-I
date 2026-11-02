#!/usr/bin/env python3
"""Package the tested WSL source snapshot for Colab, without weights or datasets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXTRA_SOURCE = [
    "scripts/colab_single_image.py", "scripts/audit_single_image_checkpoint.py",
    "scripts/validate_colab_single_image.py", "scripts/build_colab_bundle.py",
    "notebooks/SAM3I_single_image_colab.ipynb", "docs/colab_single_image.md",
    "scripts/colab_notebook_helpers.py", "scripts/single_image_visualization.py",
    "scripts/sync_colab_notebook.py", "scripts/validate_colab_github.py",
]


def build_bundle(output):
    tracked = subprocess.check_output([
        "git", "-C", str(ROOT), "ls-files", "-z", "--",
        "sam3", "scripts", "README.md", "pyproject.toml", "uv.lock", ".python-version",
    ]).decode().split("\0")
    names = sorted(set(filter(None, tracked)) | set(EXTRA_SOURCE))
    files = {}
    for name in names:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Required source missing: {name}")
        if path.suffix in {".pt", ".pth", ".pyc"} or any(
            part in {".git", ".venv", "__pycache__", ".cache"} for part in path.parts):
            raise ValueError(f"Refusing non-source artifact: {name}")
        files[name] = path.read_bytes()
    manifest = {
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "description": "Working-tree source snapshot; per-file hashes identify local changes.",
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "excludes": ["weights", "datasets", "inference outputs", "environments", ".git"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            item = zipfile.ZipInfo(f"SAM3-I/{name}", date_time=(2026, 11, 2, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, data)
        item = zipfile.ZipInfo("SAM3-I/SOURCE_MANIFEST.json", date_time=(2026, 11, 2, 0, 0, 0))
        item.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(item, json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({
        "bundle": str(output.resolve()), "source_files": len(files),
        "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/colab/sam3i-colab-source.zip")
    args = parser.parse_args()
    build_bundle(args.output)


if __name__ == "__main__":
    main()
