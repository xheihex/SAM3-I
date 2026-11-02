#!/usr/bin/env python3
"""Check a SAM3-I checkpoint against the real final-stage model on meta only.

This checks names and shapes, never checkpoint tensor values or model accuracy.
Run with CUDA_VISIBLE_DEVICES= and the project's uv environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BPE = PROJECT_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_CHECKPOINT = Path("/home/kmg/models/sam3-i/checkpoint.pt")
ADAPTER_CONFIG = {"adapter_dim": 64, "adapter_heads": 4, "adapter_scale": 1.0}


def _adapter_group(key: str) -> str:
    for name in ("adapter12", "adapter1", "adapter2"):
        if name in key:
            return name
    return "other_adapter"


def audit_checkpoint(checkpoint_path: Path, bpe_path: Path) -> dict:
    """Return a shape audit; ``ok`` is false for missing/mismatched state keys.

    CUDA visibility must be explicitly disabled before the process starts. The
    position-encoding cache is disabled only within this audit: upstream eagerly
    creates its optional cache on CUDA even when the model device is CPU/meta.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run in a fresh process with CUDA_VISIBLE_DEVICES= (empty).")

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was already initialized; use a fresh CPU-only process.")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    bpe_path = Path(bpe_path).expanduser().resolve(strict=True)
    package_root = str(PROJECT_ROOT / "sam3")
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    import sam3.model_builder as builder

    if torch.cuda.is_initialized():
        raise RuntimeError("Import initialized CUDA unexpectedly.")
    original_position_encoding = builder._create_position_encoding

    def position_encoding_without_cache(precompute_resolution=None):
        return original_position_encoding(precompute_resolution=None)

    # ViT converts 32 drop-path schedule scalars to Python via .item().
    # Those configuration scalars need numeric CPU storage even for meta builds.
    original_linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)

    # The decoder also eagerly requests a CUDA coordinate cache. Preserve its
    # exact shapes as meta tensors for the audit without any GPU allocation.
    original_get_coords = builder.TransformerDecoder._get_coords

    def meta_get_coords(height, width, device):
        return original_get_coords(height, width, device="meta")

    build_log = io.StringIO()
    with redirect_stdout(build_log), patch.object(
        builder, "_create_position_encoding", position_encoding_without_cache
    ), patch.object(torch, "linspace", cpu_linspace), patch.object(
        builder.TransformerDecoder, "_get_coords", staticmethod(meta_get_coords)
    ), torch.device("meta"):
        model = builder.build_sam3_image_model(
            checkpoint_path=None,
            bpe_path=str(bpe_path),
            load_from_HF=False,
            device="meta",
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            inst_stage="3",
            adapter_config=ADAPTER_CONFIG,
        )
    expected = model.state_dict()
    active_parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # The real local-checkpoint branch calls this replacement helper. It assigns
    # backbone.text, while forward_text reads backbone.language_backbone. Retain
    # the author's redundant module and audit that loading topology separately.
    with redirect_stdout(build_log), torch.device("meta"):
        builder._replace_text_encoder(model, str(bpe_path), "1_1", ADAPTER_CONFIG)
    loading_topology = model.state_dict()
    loading_parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    non_meta_expected = [key for key, value in expected.items() if value.device.type != "meta"]
    if non_meta_expected:
        raise RuntimeError(f"Expected meta tensors only: {non_meta_expected[:5]}")

    # mmap avoids allocating checkpoint storages. map_location='meta' keeps every
    # model and optimizer tensor without numeric storage. Use only trusted local
    # author checkpoints, matching the upstream loader's weights_only=False.
    checkpoint = torch.load(
        checkpoint_path, map_location="meta", mmap=True, weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dict or contain a 'model' state dict.")
    actual = checkpoint.get("model", checkpoint)
    if not isinstance(actual, dict) or not all(isinstance(key, str) for key in actual):
        raise TypeError("Checkpoint model state must be a string-keyed dict.")
    invalid_values = [key for key, value in actual.items() if not isinstance(value, torch.Tensor)]
    if invalid_values:
        raise TypeError(f"Checkpoint model state has non-tensor values: {invalid_values[:5]}")
    non_meta_actual = [key for key, value in actual.items() if value.device.type != "meta"]
    if non_meta_actual:
        raise RuntimeError(f"Checkpoint tensors were materialized: {non_meta_actual[:5]}")

    expected_keys, actual_keys = set(expected), set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatches = [
        {"key": key, "expected": list(expected[key].shape), "checkpoint": list(actual[key].shape)}
        for key in sorted(expected_keys & actual_keys)
        if expected[key].shape != actual[key].shape
    ]
    matched = len(expected_keys & actual_keys) - len(shape_mismatches)
    loading_keys = set(loading_topology)
    loading_missing = sorted(loading_keys - actual_keys)
    loading_unexpected = sorted(actual_keys - loading_keys)
    loading_shape_mismatches = [
        {"key": key, "expected": list(loading_topology[key].shape), "checkpoint": list(actual[key].shape)}
        for key in sorted(loading_keys & actual_keys)
        if loading_topology[key].shape != actual[key].shape
    ]
    expected_adapter_keys = [key for key in expected if "adapter" in key]
    actual_adapter_keys = [key for key in actual if "adapter" in key]
    if torch.cuda.is_initialized():
        raise RuntimeError("Audit initialized CUDA unexpectedly.")
    return {
        "ok": not missing and not shape_mismatches and not loading_missing and not loading_shape_mismatches,
        "validation_kind": "real-builder-meta-and-checkpoint-names-shapes-only",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        "inst_stage": "3",
        "adapter_config": ADAPTER_CONFIG,
        "upstream_files_modified": False,
        "position_encoding_precompute_disabled_for_audit_only": True,
        "drop_path_schedule_created_on_cpu_for_audit_only": True,
        "decoder_coordinate_cache_created_on_meta_for_audit_only": True,
        "checkpoint_values_loaded": False,
        "numerical_forward_executed": False,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_initialized_before": False,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "expected_state_keys": len(expected),
        "checkpoint_state_keys": len(actual),
        "matched_state_keys": matched,
        "success_criteria": "all forward-model and author-loading-topology state keys have matching names and shapes; extra checkpoint keys are reported",
        "expected_tensor_bytes": sum(value.numel() * value.element_size() for value in expected.values()),
        "forward_model_parameter_bytes": active_parameter_bytes,
        "real_checkpoint_loading_topology": {
            "explanation": "Author loader also registers backbone.text; forward_text uses backbone.language_backbone. Extra text module is retained, matching author behavior.",
            "state_keys": len(loading_topology),
            "matched_state_keys": len(loading_keys & actual_keys) - len(loading_shape_mismatches),
            "missing_keys": loading_missing,
            "shape_mismatches": loading_shape_mismatches,
            "unexpected_checkpoint_keys": loading_unexpected,
            "parameter_bytes_fp32_with_complex64_buffers_excluded": loading_parameter_bytes,
            "state_tensor_bytes": sum(value.numel() * value.element_size() for value in loading_topology.values()),
            "parameter_gib": loading_parameter_bytes / 1024 ** 3,
            "redundant_backbone_text_state_keys": len(loading_keys - expected_keys),
            "redundant_backbone_text_parameter_bytes": loading_parameter_bytes - active_parameter_bytes,
        },
        "checkpoint_tensor_bytes": sum(value.numel() * value.element_size() for value in actual.values()),
        "checkpoint_dtype_counts": dict(Counter(str(value.dtype) for value in actual.values())),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "adapter_groups_expected": dict(Counter(_adapter_group(key) for key in expected_adapter_keys)),
        "adapter_groups_checkpoint": dict(Counter(_adapter_group(key) for key in actual_adapter_keys)),
        "adapter_missing_keys": [key for key in missing if "adapter" in key],
        "adapter_unexpected_keys": [key for key in unexpected if "adapter" in key],
        "builder_log": build_log.getvalue().splitlines(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bpe", type=Path, default=DEFAULT_BPE)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    report = audit_checkpoint(args.checkpoint, args.bpe)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
