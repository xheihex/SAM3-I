"""Colab deployment helpers, mirrored into the standalone notebook."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


def run_logged(command, log_path, *, env=None, cwd=None, echo=True):
    """Print subprocess output into the notebook and keep the same text on disk."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps([str(arg) for arg in command], ensure_ascii=False) + "\n")
        log.flush()
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, cwd=cwd,
        ) as process:
            for line in process.stdout:
                if echo:
                    print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    if returncode:
        print(f"Command failed; log: {log_path}", flush=True)
        raise subprocess.CalledProcessError(returncode, command)
    return log_path


def checkout_github(repo_url, ref, destination, logs_dir):
    """Download a branch/tag/commit into a fresh, detached Git checkout."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Use a new checkout directory: {destination}")
    if not ref.strip() or ref.startswith("-"):
        raise ValueError("Git ref must be a nonempty branch, tag, or commit.")
    logs_dir = Path(logs_dir)
    run_logged(["git", "init", str(destination)], logs_dir / "git-init.log")
    run_logged(["git", "-C", str(destination), "remote", "add", "origin", repo_url],
               logs_dir / "git-remote.log")
    run_logged(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", ref],
               logs_dir / "git-fetch.log")
    run_logged(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
               logs_dir / "git-checkout.log")
    commit = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True,
    ).strip()
    required = ("pyproject.toml", "uv.lock", ".python-version",
                "scripts/colab_single_image.py", "scripts/audit_single_image_checkpoint.py",
                "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    for name in required:
        if not (destination / name).is_file():
            raise FileNotFoundError(f"GitHub revision lacks required file: {name}")
    source = {
        "repository": repo_url, "requested_ref": ref, "git_commit": commit,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_file_sha256": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in required
        },
    }
    print(f"GitHub revision: {commit}", flush=True)
    return destination, source


def ensure_local_checkpoint(source, destination):
    """Reuse a complete local copy only while the source metadata still matches."""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    source_stat = source.stat()
    fingerprint = {
        "source_path": str(source.resolve()), "size_bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
    }
    if source.resolve() == destination.resolve():
        return fingerprint
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_suffix(destination.suffix + ".source.json")
    if destination.is_file() and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text())
        if previous == fingerprint and destination.stat().st_size == source_stat.st_size:
            print("Reusing local checkpoint; Drive source metadata is unchanged.", flush=True)
            return fingerprint
    if shutil.disk_usage(destination.parent).free < source_stat.st_size + 1024**3:
        raise RuntimeError("Insufficient local disk space for checkpoint copy + 1 GiB.")
    partial = destination.with_suffix(destination.suffix + ".part")
    print("Copying checkpoint from Drive to local disk...", flush=True)
    shutil.copy2(source, partial)
    if partial.stat().st_size != source_stat.st_size:
        raise RuntimeError("Incomplete local checkpoint copy.")
    partial.replace(destination)
    metadata_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return fingerprint


def render_saved_results(output_dir, image_path, source, overlay_renderer, figure_renderer):
    """Add stable IDs and presentation artifacts to old or new inference outputs."""
    import numpy as np
    from PIL import Image

    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text())
    predictions = result["predictions"]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    if (result["width"], result["height"]) != image.size:
        raise ValueError("Result dimensions do not match the uploaded image.")
    masks = np.empty((len(predictions), image.height, image.width), dtype=bool)
    for index, prediction in enumerate(predictions):
        # Keep the saved prediction order and mask filenames; only add one-based IDs.
        prediction["instance_id"] = index + 1
        mask_path = (output_dir / prediction["mask_file"]).resolve()
        if not mask_path.is_relative_to(output_dir.resolve()):
            raise ValueError("Mask file must be inside the result directory.")
        with Image.open(mask_path) as opened_mask:
            if opened_mask.size != image.size:
                raise ValueError("Mask dimensions do not match the uploaded image.")
            masks[index] = np.asarray(opened_mask.convert("L")) > 0
    overlay = overlay_renderer(image, masks, predictions)
    overlay.save(output_dir / "overlay.png")
    figure = figure_renderer(overlay, predictions)
    figure.save(output_dir / "result_figure.png")
    result["source"] = source
    result["visualization"] = {
        "instance_id_start": 1, "overlay": "overlay.png", "figure": "result_figure.png",
        "description": "Colored masks, boxes and IDs; confidence scores below the image.",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, figure


def archive_run(output_dir, source, audit_path, logs_dir, environment_path, drive_results):
    """Save provenance and logs with results, then immediately persist a ZIP to Drive."""
    output_dir, drive_results = Path(output_dir), Path(drive_results)
    (output_dir / "source-manifest.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    shutil.copy2(audit_path, output_dir / "checkpoint-audit.json")
    shutil.copy2(environment_path, output_dir / "environment.json")
    shutil.copytree(logs_dir, output_dir / "logs", dirs_exist_ok=True)
    archive = Path(shutil.make_archive(str(output_dir), "zip", root_dir=output_dir))
    drive_results.mkdir(parents=True, exist_ok=True)
    saved_archive = drive_results / archive.name
    shutil.copy2(archive, saved_archive)
    print(f"Saved to Drive: {saved_archive}", flush=True)
    return archive, saved_archive
