#!/usr/bin/env python3
"""CPU checks for GitHub Colab deployment, result numbering and Drive persistence."""
from __future__ import annotations
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image
import torch
import colab_single_image as single
import colab_notebook_helpers as helpers
import single_image_visualization as visual
from sync_colab_notebook import embedded_source


def old_format_results(destination, *, empty=False):
    """Use the unchanged inference exporter, substituting only model outputs."""
    image = Image.new("RGB", (320, 240), (95, 115, 135))
    masks = torch.zeros((0 if empty else 2, 1, 240, 320), dtype=torch.bool)
    boxes = torch.empty((0, 4)) if empty else torch.tensor([[20, 30, 100, 140], [160, 80, 290, 210]])
    if not empty:
        masks[0, 0, 30:140, 20:100] = True
        masks[1, 0, 80:210, 160:290] = True
    state = {"masks": masks, "boxes": boxes, "scores": torch.empty(0) if empty else torch.tensor([0.93, 0.76])}
    single.save_results(image, state, destination, {
        "synthetic": True, "precision": "fp32", "category": "simple",
        "threshold": 0.6, "elapsed_seconds": 1.25, "peak_cuda_allocated_gib": 0.0,
    })
    return image


class GithubNotebookTests(unittest.TestCase):
    materials = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=self.materials)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_embedded_sources_syntax_and_no_outputs(self):
        notebook = json.loads((ROOT / "notebooks/SAM3I_single_image_colab.ipynb").read_text())
        embedded = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            ast.parse(source, filename=f"notebook-cell-{index}")
            self.assertFalse(cell.get("outputs"))
            self.assertIsNone(cell.get("execution_count"))
            if name := cell.get("metadata", {}).get("sam3i_source"):
                self.assertEqual(source, embedded_source(name))
                embedded.append(name)
        self.assertEqual(set(embedded), {"scripts/colab_notebook_helpers.py", "scripts/single_image_visualization.py"})

    def test_subprocess_logs_stdout_stderr_and_failure(self):
        log = self.folder / "stdout-stderr.log"
        with redirect_stdout(io.StringIO()) as console:
            helpers.run_logged([sys.executable, "-c", "import sys; print('stdout-marker'); print('stderr-marker',file=sys.stderr)"], log)
        self.assertIn("stdout-marker", console.getvalue())
        self.assertIn("stderr-marker", console.getvalue())
        self.assertIn("stderr-marker", log.read_text())
        failed_log = self.folder / "failed.log"
        with self.assertRaises(subprocess.CalledProcessError):
            with redirect_stdout(io.StringIO()):
                helpers.run_logged([sys.executable, "-c", "print('failure-marker'); raise SystemExit(3)"], failed_log)
        self.assertIn("failure-marker", failed_log.read_text())

    def test_checkpoint_reuse_and_source_change(self):
        source = self.folder / "Drive/source.pt"
        source.parent.mkdir()
        source.write_bytes(b"first")
        target = self.folder / "local/checkpoint.pt"
        with redirect_stdout(io.StringIO()):
            first = helpers.ensure_local_checkpoint(source, target)
            with patch.object(helpers.shutil, "copy2", side_effect=AssertionError("should reuse")):
                second = helpers.ensure_local_checkpoint(source, target)
            self.assertEqual(first, second)
            source.write_bytes(b"other")
            os.utime(source, ns=(source.stat().st_atime_ns, first["mtime_ns"] + 10000000))
            helpers.ensure_local_checkpoint(source, target)
        self.assertEqual(target.read_bytes(), b"other")
        self.assertFalse(target.with_suffix(".pt.part").exists())

    def test_old_schema_numbering_masks_and_empty(self):
        for empty in (False, True):
            output = self.folder / str(empty)
            image = old_format_results(output, empty=empty)
            image_path = self.folder / f"image-{empty}.png"
            image.save(image_path)
            original = json.loads((output / "result.json").read_text())
            result, figure = helpers.render_saved_results(
                output, image_path, {"git_commit": "synthetic"}, visual.render_overlay, visual.render_result_figure)
            self.assertEqual([p["instance_id"] for p in result["predictions"]],
                             list(range(1, len(original["predictions"]) + 1)))
            self.assertEqual([p["mask_file"] for p in result["predictions"]],
                             [p["mask_file"] for p in original["predictions"]])
            self.assertEqual([p["score"] for p in result["predictions"]],
                             [p["score"] for p in original["predictions"]])
            self.assertGreater(figure.height, image.height)
            self.assertEqual(Image.open(output / "overlay.png").size, image.size)

    def test_invalid_numbering_and_scores_rejected(self):
        prediction = {"instance_id": 1, "score": 0.9, "box_xyxy": [0, 0, 40, 30]}
        with self.assertRaises(ValueError):
            visual.render_result_figure(Image.new("RGB", (100, 80)), [prediction, prediction])
        with self.assertRaises(ValueError):
            visual.render_result_figure(Image.new("RGB", (100, 80)), [{**prediction, "score": float("nan")}])

    def test_actual_inference_cell_saves_without_download_cell(self):
        notebook = json.loads((ROOT / "notebooks/SAM3I_single_image_colab.ipynb").read_text())
        namespace = {"__name__": "notebook_cpu_check"}
        for cell in notebook["cells"]:
            if cell.get("metadata", {}).get("sam3i_source"):
                exec(compile("".join(cell["source"]), "<embedded>", "exec"), namespace)
        image_path = self.folder / "input.png"
        Image.new("RGB", (320, 240), (95, 115, 135)).save(image_path)
        logs = self.folder / "logs"
        logs.mkdir()
        environment = self.folder / "environment.json"
        environment.write_text(json.dumps({"gpu": "synthetic-no-GPU"}))
        audit = self.folder / "audit.json"
        audit.write_text('{"ok":true,"synthetic":true}')
        def fake_inference(command, log_path, **kwargs):
            # Exercise the exact notebook cell while replacing only GPU computation.
            self.assertIn("--checkpoint", command)
            self.assertIn("--category", command)
            destination = Path(command[command.index("--output-dir") + 1])
            old_format_results(destination)
            Path(log_path).write_text("SYNTHETIC MODEL OUTPUT - no GPU used\n")
        real_path = Path
        def mapped_path(path):
            if str(path) == "/content/sam3i-results":
                return self.folder / "results"
            return real_path(path)
        displayed = []
        namespace.update(
            Path=mapped_path, uuid=uuid, REPO=ROOT, RUN=["synthetic-python"], RUN_ENV={},
            LOGS=logs, CHECKPOINT=self.folder / "checkpoint.pt", IMAGE_PATH=image_path,
            SOURCE_INFO={"git_commit": "synthetic"}, ENVIRONMENT_PATH=environment,
            AUDIT_PATH=audit, DRIVE_RESULTS=self.folder / "Drive-results",
            run_logged=fake_inference, display=lambda picture: displayed.append(picture.copy()),
        )
        cell = next(cell for cell in notebook["cells"]
                    if cell["cell_type"] == "code" and "PROMPT = " in "".join(cell["source"]))
        with redirect_stdout(io.StringIO()):
            exec(compile("".join(cell["source"]), "<actual-inference-cell>", "exec"), namespace)
        self.assertEqual(len(displayed), 1)
        saved = namespace["SAVED_RESULT"]
        self.assertTrue(saved.is_file(), "Drive archive must exist before optional download cell")
        with zipfile.ZipFile(saved) as archive:
            for required in ("result_figure.png", "overlay.png", "result.json",
                             "source-manifest.json", "environment.json", "checkpoint-audit.json"):
                self.assertIn(required, archive.namelist())
            self.assertTrue(any(name.startswith("logs/inference-") for name in archive.namelist()))
            result = json.loads(archive.read("result.json"))
        self.assertEqual([p["instance_id"] for p in result["predictions"]], [1, 2])
        # Leave one explicitly synthetic inspection artifact, with scores shown below.
        displayed[0].save(self.materials / "notebook-cell-synthetic-figure.png")


def github_smoke(materials):
    checkout_base = ROOT / ".cache" / f"colab-github-source-check-{uuid.uuid4().hex[:12]}"
    logs = materials / "github-smoke-logs"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    uv = str(Path(sys.executable).parent / "uv")
    if not Path(uv).is_file():
        uv = shutil.which("uv") or "/home/kmg/.local/bin/uv"
    try:
        project, source = helpers.checkout_github(
            "https://github.com/xheihex/SAM3-I.git", "main", checkout_base / "SAM3-I", logs)
        helpers.run_logged([uv, "sync", "--locked", "--no-dev", "--offline", "--project", str(project)],
                           logs / "uv-sync.log", env=env, echo=False)
        helpers.run_logged([uv, "run", "--locked", "--no-dev", "--offline", "--project", str(project),
                            "python", str(project / "scripts/colab_single_image.py"), "--help"],
                           logs / "inference-help.log", env=env, echo=False)
        source["locked_environment_recreated_offline"] = True
        source["inference_cli_help_ok"] = True
        source["real_model_forward_executed"] = False
        (materials / "github-smoke.json").write_text(json.dumps(source, indent=2))
        return source
    finally:
        resolved = checkout_base.resolve()
        assert resolved.parent == (ROOT / ".cache").resolve()
        assert resolved.name.startswith("colab-github-source-check-")
        if resolved.exists():
            shutil.rmtree(resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--github-smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    GithubNotebookTests.materials = args.output_dir.resolve()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must remain uninitialized.")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GithubNotebookTests)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        smoke = github_smoke(GithubNotebookTests.materials) if args.github_smoke and result.wasSuccessful() else None
    report = {
        "success": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "cuda_initialized_before": False, "cuda_initialized_after": torch.cuda.is_initialized(),
        "gpu_model_loaded": False,
        "notebook_inference_cell": "Executed with real exporter and synthetic model outputs.",
        "github_smoke": smoke,
    }
    (args.output_dir / "github-notebook-check.json").write_text(json.dumps(report, indent=2))
    if not result.wasSuccessful() or torch.cuda.is_initialized():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
