#!/usr/bin/env python3
"""Persistent SAM3-I model host for the Colab adapter; no upstream changes."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import torch

# Filled after --repo-root is parsed when launched outside the checkout.
single = None


class ResidentSession:
    """Keep only the eval model between requests, never an image/prompt state."""

    def __init__(self, checkpoint, bpe):
        self.checkpoint, self.bpe = Path(checkpoint), Path(bpe)
        self.worker_id = uuid.uuid4().hex
        self.query_index = 0
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a CUDA GPU runtime before loading the model.")
        self.native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        started = time.perf_counter()
        self.model = single.build_model(self.checkpoint, self.bpe)
        torch.cuda.synchronize()
        self.model_load_seconds = time.perf_counter() - started
        self.info = {
            "worker_id": self.worker_id, "worker_pid": os.getpid(),
            "model_load_seconds": self.model_load_seconds,
            "checkpoint": str(self.checkpoint), "gpu": torch.cuda.get_device_name(0),
            "native_bf16": self.native_bf16, "torch": torch.__version__,
        }

    def run(self, request):
        image_path = Path(request["image"])
        prompt, category = request["prompt"], request["category"]
        threshold = float(request["threshold"])
        single.validate_request(image_path, prompt, category, threshold)
        requested = request.get("precision", "auto")
        if requested not in ("auto", "bf16", "fp16", "fp32"):
            raise ValueError("Unknown precision.")
        precision = single.choose_precision(requested, self.native_bf16)
        output = Path(request["output_dir"])
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Output directory is not empty; use a new directory per query.")
        image = single.open_rgb(image_path)
        self.query_index += 1
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        # Fresh processor/image/text state per query; only self.model stays resident.
        state = single.predict(
            self.model, image, prompt, category, threshold,
            device="cuda", precision=precision,
        )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        result = single.save_results(image, state, output, {
            "image_file": str(image_path), "prompt": prompt, "category": category,
            "text_type": single.TEXT_TYPES[category], "threshold": threshold,
            "precision": precision, "checkpoint": str(self.checkpoint), "inst_stage": "3",
            "worker_id": self.worker_id, "worker_pid": os.getpid(),
            "query_index": self.query_index, "model_load_seconds": self.model_load_seconds,
            "inference_seconds": inference_seconds, "elapsed_seconds": inference_seconds,
            "timing_scope": "Per-query predict including CPU mask resize; excludes model load, export and display.",
            "peak_cuda_allocated_gib": peak_gib,
        })
        # Do not return state/backbone_out: these contain GPU tensors.
        return result


def write_reply(ipc_dir, name, reply):
    destination = ipc_dir / (name + ".json")
    partial = destination.with_suffix(".tmp")
    partial.write_text(json.dumps(reply, ensure_ascii=False), encoding="utf-8")
    partial.replace(destination)


def serve(session, ipc_dir):
    write_reply(ipc_dir, "ready", {"ok": True, "result": session.info})
    print(json.dumps({"model_ready": session.info}, ensure_ascii=False), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        request_id = request.pop("id")
        if not isinstance(request_id, str) or len(request_id) != 32:
            raise ValueError("Invalid request ID.")
        int(request_id, 16)
        print(json.dumps({"request_id": request_id, **request}, ensure_ascii=False), flush=True)
        try:
            result = session.run(request)
        except Exception as error:
            # CUDA runtime/OOM errors invalidate this host. Validation/I/O errors
            # can be corrected without paying for another model load.
            fatal = isinstance(error, (torch.OutOfMemoryError, RuntimeError))
            reply = {
                "ok": False, "error": str(error), "error_type": type(error).__name__,
                "fatal": fatal,
            }
            traceback.print_exc()
        else:
            reply = {"ok": True, "result": result}
        # Exception traceback and run() locals are out of scope here.
        if not reply["ok"] and not reply.get("fatal"):
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                # A broken CUDA context must not hide the original request error.
                traceback.print_exc()
                reply["fatal"] = True
        write_reply(ipc_dir, request_id, reply)
        print(f"Request {request_id}: {'ok' if reply['ok'] else 'failed'}", flush=True)
        if reply.get("fatal"):
            return


def main():
    global single
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe", type=Path)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root / "scripts"))
    import colab_single_image as adapter
    single = adapter
    args.ipc_dir.mkdir(parents=True, exist_ok=True)
    session = ResidentSession(args.checkpoint, args.bpe or single.DEFAULT_BPE)
    serve(session, args.ipc_dir)


if __name__ == "__main__":
    main()
