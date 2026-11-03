#!/usr/bin/env python3
"""CPU checks for model reuse and real subprocess IPC; no GPU inference."""
from __future__ import annotations
import argparse
from contextlib import ExitStack, redirect_stdout
import gc
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from PIL import Image
import torch
import colab_single_image as single
import colab_inference_worker as worker
import colab_notebook_helpers as helpers

FAKE_WORKER = r"""
import json, os, sys, time
from pathlib import Path
folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
def reply(name, value):
    path = folder / name
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(value))
    partial.replace(path)
reply("ready.json", {"ok": True, "result": {
    "worker_id": "synthetic-worker", "worker_pid": os.getpid(),
    "model_load_seconds": 0.0,
}})
print("synthetic worker ready", flush=True)
count = 0
for line in sys.stdin:
    request = json.loads(line)
    if request.get("command") == "stop" or request.get("op") == "stop":
        break
    action = request.get("prompt")
    if action == "die":
        print("synthetic worker exiting", flush=True)
        raise SystemExit(7)
    if action == "hang":
        time.sleep(30)
    if action == "error":
        reply(request["id"] + ".json", {
            "ok": False, "error": "synthetic request error", "error_type": "ValueError",
        })
        continue
    count += 1
    print("synthetic query " + str(count), flush=True)
    reply(request["id"] + ".json", {"ok": True, "result": {
        "worker_id": "synthetic-worker", "worker_pid": os.getpid(),
        "query_index": count, "prompt": action, "synthetic": True,
    }})
"""

class ResidentTests(unittest.TestCase):
    materials = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=self.materials)
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.image = self.folder / "input.png"
        Image.new("RGB", (32, 24), (90, 110, 130)).save(self.image)
        self.checkpoint = self.folder / "checkpoint.pt"
        self.checkpoint.write_bytes(b"synthetic: not torch data")
        self.bpe = self.folder / "bpe.gz"
        self.bpe.write_bytes(b"synthetic")
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(worker, "single", single))
        for name, value in (
            ("is_available", True), ("is_bf16_supported", True),
            ("synchronize", None), ("reset_peak_memory_stats", None),
            ("max_memory_allocated", 1024), ("get_device_name", "synthetic CPU GPU"),
        ):
            stack.enter_context(patch.object(torch.cuda, name, return_value=value))
        stack.enter_context(patch.object(torch.cuda, "empty_cache"))

    def request(self, name="one", **overrides):
        return {
            "image": str(self.image), "prompt": "shoe", "category": "simple",
            "threshold": 0.5, "precision": "fp32",
            "output_dir": str(self.folder / name), **overrides,
        }

    def predictions(self, model, image, prompt, category, threshold, **kwargs):
        self.calls.append({
            "model": model, "size": image.size, "prompt": prompt,
            "category": category, "threshold": threshold, **kwargs,
        })
        masks = torch.zeros((1, 1, image.height, image.width), dtype=torch.bool)
        masks[0, 0, 4:12, 5:15] = True
        self.references.append(weakref.ref(masks))
        return {
            "masks": masks, "boxes": torch.tensor([[5., 4., 15., 12.]]),
            "scores": torch.tensor([0.9]),
        }

    def session(self):
        self.calls, self.references = [], []
        self.model = object()
        self.builder = self.enterContext(patch.object(single, "build_model", return_value=self.model))
        self.predictor = self.enterContext(patch.object(single, "predict", side_effect=self.predictions))
        with redirect_stdout(io.StringIO()):
            return worker.ResidentSession(self.checkpoint, self.bpe)

    def test_model_reuse_changed_inputs_and_request_tensor_release(self):
        session = self.session()
        first = session.run(self.request())
        other = self.folder / "second.png"
        Image.new("RGB", (48, 40)).save(other)
        second = session.run(self.request(
            "two", image=str(other), prompt="red item", category="complex",
            threshold=0.7, precision="auto",
        ))
        self.builder.assert_called_once()
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(call["model"] is self.model for call in self.calls))
        self.assertEqual([call["size"] for call in self.calls], [(32, 24), (48, 40)])
        self.assertEqual(self.calls[1]["category"], "complex")
        self.assertEqual(self.calls[1]["threshold"], 0.7)
        self.assertEqual(self.calls[1]["precision"], "bf16")
        self.assertEqual(first["worker_id"], second["worker_id"])
        self.assertEqual(first["worker_pid"], second["worker_pid"])
        self.assertEqual(second["query_index"], first["query_index"] + 1)
        for result in (first, second):
            self.assertEqual(result["num_instances"], 1)
            for name in ("model_load_seconds", "inference_seconds", "elapsed_seconds",
                         "peak_cuda_allocated_gib"):
                self.assertGreaterEqual(result[name], 0)
        for name in ("one", "two"):
            saved = json.loads((self.folder / name / "result.json").read_text())
            self.assertEqual(saved["worker_id"], first["worker_id"])
            self.assertTrue((self.folder / name / "mask_000.png").is_file())
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.references),
                        "Session must release per-request output tensors.")

    def test_invalid_request_and_collision_recover_without_reload(self):
        session = self.session()
        for overrides in (
            {"prompt": " "}, {"threshold": float("nan")}, {"threshold": -0.1},
            {"category": "unknown"}, {"image": str(self.folder / "missing.png")},
            {"precision": "unknown"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    session.run(self.request(**overrides))
        self.assertEqual(self.predictor.call_count, 0)
        session.run(self.request("valid"))
        saved = (self.folder / "valid/result.json").read_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            session.run(self.request("valid"))
        self.assertEqual(saved, (self.folder / "valid/result.json").read_bytes())
        session.run(self.request("after-error", category="concept"))
        self.assertEqual(self.predictor.call_count, 2)
        self.builder.assert_called_once()

    def test_actual_serve_recovers_validation_and_invalidates_runtime_error(self):
        session = self.session()
        def predict(*args, **kwargs):
            if args[2] == "crash":
                raise RuntimeError("synthetic fatal CUDA error")
            return self.predictions(*args, **kwargs)
        self.predictor.side_effect = predict
        requests = [
            {"id": "a" * 32, **self.request("invalid", prompt=" ")},
            {"id": "b" * 32, **self.request("valid")},
            {"id": "c" * 32, **self.request("fatal", prompt="crash")},
            {"id": "d" * 32, **self.request("not-run")},
        ]
        ipc = self.folder / "serve-ipc"
        ipc.mkdir()
        from contextlib import redirect_stderr
        with patch.object(worker.sys, "stdin", io.StringIO(
            "".join(json.dumps(request) + "\n" for request in requests)
        )), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            worker.serve(session, ipc)
        invalid = json.loads((ipc / ("a" * 32 + ".json")).read_text())
        valid = json.loads((ipc / ("b" * 32 + ".json")).read_text())
        fatal = json.loads((ipc / ("c" * 32 + ".json")).read_text())
        self.assertFalse(invalid["ok"])
        self.assertFalse(invalid["fatal"])
        self.assertTrue(valid["ok"])
        self.assertEqual(valid["result"]["num_instances"], 1)
        self.assertFalse(fatal["ok"])
        self.assertTrue(fatal["fatal"])
        self.assertIn("synthetic fatal CUDA error", fatal["error"])
        self.assertFalse((ipc / ("d" * 32 + ".json")).exists())
        self.builder.assert_called_once()

        cleanup_ipc = self.folder / "cleanup-failure-ipc"
        cleanup_ipc.mkdir()
        cleanup_requests = [
            {"id": "e" * 32, **self.request("cleanup-invalid", prompt=" ")},
            {"id": "f" * 32, **self.request("cleanup-not-run")},
        ]
        diagnostics = io.StringIO()
        with patch.object(worker.sys, "stdin", io.StringIO(
            "".join(json.dumps(request) + "\n" for request in cleanup_requests)
        )), patch.object(torch.cuda, "empty_cache",
                         side_effect=RuntimeError("synthetic broken CUDA cleanup")), \
                redirect_stdout(io.StringIO()), redirect_stderr(diagnostics):
            worker.serve(session, cleanup_ipc)
        cleanup_reply = json.loads((cleanup_ipc / ("e" * 32 + ".json")).read_text())
        self.assertFalse(cleanup_reply["ok"])
        self.assertTrue(cleanup_reply["fatal"])
        self.assertEqual(cleanup_reply["error_type"], "ValueError")
        self.assertIn("Prompt must be a nonempty string", cleanup_reply["error"])
        self.assertNotIn("broken CUDA cleanup", cleanup_reply["error"])
        self.assertIn("synthetic broken CUDA cleanup", diagnostics.getvalue())
        self.assertFalse((cleanup_ipc / ("f" * 32 + ".json")).exists())
        self.builder.assert_called_once()

    def test_start_model_worker_reuses_and_replaces_changed_configuration(self):
        repo = self.folder / "repo"
        for name in ("sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                     "scripts/colab_single_image.py", ".venv/bin/python"):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic " + name)
        events = []
        clients = []
        class FakeClient:
            def __init__(self, command, ipc_dir, log_path, **kwargs):
                events.append("construct")
                self.alive = True
                self.info = {"worker_id": str(len(clients))}
                clients.append(self)
            def is_alive(self):
                return self.alive
            def stop(self):
                events.append("stop")
                self.alive = False
        def start(existing=None, source="first worker"):
            with redirect_stdout(io.StringIO()):
                return helpers.start_model_worker(
                    repo, self.checkpoint, self.folder / "runtime",
                    self.folder / "logs", existing=existing, worker_source=source,
                )
        with patch.object(helpers, "ResidentInferenceWorker", FakeClient):
            first = start()
            self.assertIs(start(first), first)
            self.assertEqual(events, ["construct"])
            first.alive = False
            second = start(first)
            self.assertIsNot(second, first)
            stat = self.checkpoint.stat()
            os.utime(self.checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
            third = start(second)
            fourth = start(third, source="changed worker")
            (repo / "scripts/colab_single_image.py").write_text("changed adapter")
            fifth = start(fourth, source="changed worker")
        self.assertEqual(events, ["construct", "stop", "construct", "stop",
                                  "construct", "stop", "construct", "stop", "construct"])
        self.assertEqual(len(clients), 5)
        self.assertTrue(fifth.is_alive())
        self.assertTrue(all(not client.is_alive() for client in clients[:-1]))

    def client(self):
        script = self.folder / "fake_worker.py"
        script.write_text(FAKE_WORKER)
        ipc = self.folder / "ipc"
        with redirect_stdout(io.StringIO()):
            client = helpers.ResidentInferenceWorker(
                [sys.executable, "-u", str(script), str(ipc)],
                ipc, self.folder / "worker.log", startup_timeout=5,
            )
        self.addCleanup(client.stop)
        return client

    def test_real_ipc_reuse_error_recovery_logging_and_stop(self):
        client = self.client()
        self.assertTrue(client.is_alive())
        self.assertEqual(client.info["worker_id"], "synthetic-worker")
        first = client.predict(self.request(), log_path=self.folder / "one.log", timeout=5)
        with self.assertRaisesRegex(RuntimeError, "synthetic request error"):
            client.predict(self.request(prompt="error"), timeout=5)
        second = client.predict(self.request("two", prompt="second"), timeout=5)
        self.assertEqual(first["worker_pid"], second["worker_pid"])
        self.assertEqual(second["query_index"], 2)
        self.assertIn("synthetic", (self.folder / "one.log").read_text())
        client.stop()
        client.stop()
        self.assertFalse(client.is_alive())

    def test_real_ipc_worker_death(self):
        client = self.client()
        with self.assertRaises((RuntimeError, BrokenPipeError)):
            client.predict(self.request(prompt="die"), timeout=5)
        self.assertFalse(client.is_alive())
        with self.assertRaises((RuntimeError, BrokenPipeError)):
            client.predict(self.request(), timeout=1)

    def test_real_ipc_timeout_stops_worker(self):
        client = self.client()
        with self.assertRaises((TimeoutError, RuntimeError)):
            client.predict(self.request(prompt="hang"), timeout=0.2)
        self.assertFalse(client.is_alive(), "Timeout must not race future requests.")

    def test_failed_startup(self):
        script = self.folder / "fail.py"
        script.write_text("raise SystemExit(9)\n")
        with self.assertRaises(RuntimeError):
            helpers.ResidentInferenceWorker(
                [sys.executable, str(script)], self.folder / "failed-ipc",
                self.folder / "failed.log", startup_timeout=3,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ResidentTests.materials = args.output_dir.resolve()
    before = torch.cuda.is_initialized()
    if before:
        raise RuntimeError("CUDA must remain uninitialized during CPU validation.")
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ResidentTests))
    report = {
        "mode": "CPU synthetic model reuse and real process IPC",
        "success": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "cuda_initialized_before": before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "real_checkpoint_loaded": False, "real_gpu_forward_executed": False,
        "limitations": [
            "Model forward and CUDA statistics use synthetic CPU values.",
            "Real subprocess IPC checks do not measure GPU inference speed.",
        ],
    }
    (args.output_dir / "resident-worker-check.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if not result.wasSuccessful() or torch.cuda.is_initialized():
        raise SystemExit(1)

if __name__ == "__main__":
    main()
