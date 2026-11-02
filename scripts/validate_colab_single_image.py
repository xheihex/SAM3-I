#!/usr/bin/env python3
"""Offline CPU contract checks. Synthetic outputs are not segmentation results."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch

import colab_single_image as single
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.tokenizer_ve import SimpleTokenizer


def raw_outputs():
    masks = torch.full((1, 3, 4, 4), -8.0)
    masks[0, 0, :2, :2] = 8
    masks[0, 1, 2:, 2:] = 8
    return {
        "pred_masks": masks,
        "pred_boxes": torch.tensor([[[0.25, 0.25, 0.5, 0.5],
                                     [0.75, 0.75, 0.5, 0.5],
                                     [0.5, 0.5, 1.0, 1.0]]]),
        "pred_logits": torch.tensor([[[8.0], [4.0], [-8.0]]]),
        "presence_logit_dec": torch.tensor([[6.0]]),
    }


class SyntheticBackbone:
    def __init__(self):
        self.image_shape = None
        self.text_types = []

    def forward_image(self, image):
        assert image.device.type == "cpu"
        assert image.dtype == torch.float32
        assert tuple(image.shape) == (1, 3, 1008, 1008)
        assert -1.0 <= image.min() and image.max() <= 1.0
        self.image_shape = list(image.shape)
        return {}

    def forward_text(self, prompts, text_type, device):
        assert device == "cpu" and len(prompts) == 1
        self.text_types.append(text_type)
        return {}


class SyntheticModel:
    """Replaces only numerical model computation, not preprocessing/postprocessing."""
    inst_interactive_predictor = None

    def __init__(self, outputs=None):
        self.outputs = outputs if outputs is not None else raw_outputs()
        self.backbone = SyntheticBackbone()
        self.grounding_types = []

    def _get_dummy_prompt(self):
        return object()

    def forward_grounding(self, **kwargs):
        self.grounding_types.append(kwargs["text_type"])
        return {key: value.clone() for key, value in self.outputs.items()}


class ContractTests(unittest.TestCase):
    actual_image = None
    material_dir = None
    observed = {}

    def setUp(self):
        self.image = Image.new("RGB", (96, 64), (128, 80, 32))

    def test_real_image_preprocessing_and_tokenizer(self):
        image = single.open_rgb(self.actual_image)
        model = SyntheticModel()
        state = single.Sam3Processor(model, device="cpu").set_image(image)
        self.assertEqual((state["original_width"], state["original_height"]), image.size)
        self.assertEqual(model.backbone.image_shape, [1, 3, 1008, 1008])
        tokenizer = SimpleTokenizer(bpe_path=str(single.DEFAULT_BPE), context_length=32)
        tokens = tokenizer(["the red shoe in the top row"])
        self.assertEqual(tuple(tokens.shape), (1, 32))
        self.observed["actual_input"] = {
            "path": str(self.actual_image), "size_wh": list(image.size), "mode": image.mode,
            "preprocessed_shape": model.backbone.image_shape, "tokens_shape": list(tokens.shape),
        }

    def test_all_three_routes_reuse_image_features(self):
        model = SyntheticModel()
        processor = single.CpuMaskProcessor(model, device="cpu")
        state = processor.set_image(self.image)
        for category, text_type in single.TEXT_TYPES.items():
            state = processor.set_text_prompt(category, text_type, state)
            self.assertEqual(tuple(state["masks"].shape), (2, 1, 64, 96))
        self.assertEqual(model.backbone.text_types, [1, 2, 4])
        self.assertEqual(model.grounding_types, [1, 2, 4])

    def test_postprocessing_matches_author_fp32_and_empty(self):
        for threshold in (0.0, 0.5, 1.0):
            for text_type in (1, 2, 4):
                with self.subTest(threshold=threshold, text_type=text_type):
                    original = Sam3Processor(SyntheticModel(), device="cpu",
                                             confidence_threshold=threshold)
                    adapted = single.CpuMaskProcessor(SyntheticModel(), device="cpu",
                                                       confidence_threshold=threshold)
                    expected = original.set_text_prompt(
                        "shoe", text_type, original.set_image(self.image),
                    )
                    actual = adapted.set_text_prompt(
                        "shoe", text_type, adapted.set_image(self.image),
                    )
                    for key in ("scores", "boxes", "masks"):
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    self.assertNotIn("masks_logits", actual)

    def test_predict_and_export_rle_png_overlay_cutout(self):
        state = single.predict(SyntheticModel(), self.image, "shoe", "concept", 0.5,
                               device="cpu", precision="fp32")
        destination = self.material_dir / "synthetic-nonempty"
        result = single.save_results(self.image, state, destination, {"synthetic": True})
        self.assertEqual(result["num_instances"], 2)
        union = np.zeros((64, 96), dtype=np.bool_)
        for item in result["predictions"]:
            decoded = mask_utils.decode(item["rle"]).astype(bool)
            exported = np.array(Image.open(destination / item["mask_file"])) > 0
            np.testing.assert_array_equal(decoded, exported)
            self.assertEqual(int(exported.sum()), item["area_pixels"])
            union |= decoded
        np.testing.assert_array_equal(
            np.array(Image.open(destination / "union_mask.png")) > 0, union,
        )
        np.testing.assert_array_equal(
            np.array(Image.open(destination / "cutout.png"))[:, :, 3] > 0, union,
        )
        self.assertEqual(Image.open(destination / "overlay.png").size, self.image.size)

    def test_empty_export(self):
        state = single.predict(SyntheticModel(), self.image, "absent", "complex", 1.0,
                               device="cpu", precision="fp32")
        result = single.save_results(
            self.image, state, self.material_dir / "synthetic-empty", {"synthetic": True},
        )
        self.assertEqual(result["predictions"], [])
        self.assertEqual(tuple(state["masks"].shape), (0, 1, 64, 96))
        cutout = np.array(Image.open(self.material_dir / "synthetic-empty/cutout.png"))
        self.assertFalse(cutout[:, :, 3].any())

    def test_nonfinite_prediction_rejected(self):
        outputs = raw_outputs()
        outputs["pred_logits"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            single.predict(SyntheticModel(outputs), self.image, "shoe", "simple", 0.5,
                           device="cpu", precision="fp32")

    def test_precision_and_validation(self):
        self.assertEqual(single.choose_precision("auto", True), "bf16")
        self.assertEqual(single.choose_precision("auto", False), "fp16")
        self.assertEqual(single.choose_precision("fp32", False), "fp32")
        with self.assertRaises(ValueError):
            single.choose_precision("bf16", False)
        for threshold in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                single.validate_request(self.actual_image, "shoe", "concept", threshold)
        with self.assertRaises(ValueError):
            single.validate_request(self.actual_image, "  ", "simple", 0.5)

    def test_stage_three_and_scoped_mmap_loader(self):
        from sam3 import model_builder
        original_loader = model_builder._load_raw_checkpoint
        with patch.object(single, "build_sam3_image_model") as builder:
            def check(**kwargs):
                self.assertIs(model_builder._load_raw_checkpoint, single.mmap_checkpoint_state)
                return "model"
            builder.side_effect = check
            self.assertEqual(single.build_model("checkpoint.pt", single.DEFAULT_BPE), "model")
            kwargs = builder.call_args.kwargs
            self.assertEqual(kwargs["inst_stage"], "3")
            self.assertFalse(kwargs["load_from_HF"])
            self.assertFalse(kwargs["enable_inst_interactivity"])
            self.assertEqual(kwargs["adapter_config"], single.ADAPTER_CONFIG)
        self.assertIs(model_builder._load_raw_checkpoint, original_loader)

    def test_scoped_mmap_loader_restored_after_build_error(self):
        from sam3 import model_builder
        original_loader = model_builder._load_raw_checkpoint
        with patch.object(single, "build_sam3_image_model", side_effect=RuntimeError("test")):
            with self.assertRaisesRegex(RuntimeError, "test"):
                single.build_model("checkpoint.pt", single.DEFAULT_BPE)
        self.assertIs(model_builder._load_raw_checkpoint, original_loader)

    def test_real_mmap_loader_with_tiny_training_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=self.material_dir) as folder:
            path = Path(folder) / "tiny.pt"
            torch.save({"model": {"weight": torch.arange(4)}, "optimizer": {"unused": 1}}, path)
            result = single.mmap_checkpoint_state(path)
            self.assertEqual(set(result), {"weight"})
            torch.testing.assert_close(result["weight"], torch.arange(4))
            torch.save({"weight": torch.arange(4)}, path)
            raw_result = single.mmap_checkpoint_state(path)
            torch.testing.assert_close(raw_result["weight"], torch.arange(4))

    def test_notebook_cell_syntax(self):
        notebook = json.loads((ROOT / "notebooks/SAM3I_single_image_colab.ipynb").read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"colab-cell-{index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ContractTests.actual_image = args.image.resolve()
    ContractTests.material_dir = args.output_dir.resolve()
    before = torch.cuda.is_initialized()
    if before:
        raise RuntimeError("CUDA was already initialized before offline validation.")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA is forbidden")):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "mode": "CPU-real-preprocessing-synthetic-model-contract-checks",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "success": result.wasSuccessful(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_initialized_before": before, "cuda_initialized_after": torch.cuda.is_initialized(),
        **ContractTests.observed,
        "limitations": [
            "Synthetic masks and scores are not predictions for the real input.",
            "No real checkpoint numerical forward, CUDA kernel, or Colab execution was tested.",
            "FP32 CPU postprocessing parity does not prove BF16/FP16 GPU pixel identity.",
        ],
    }
    (args.output_dir / "cpu-contract-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if not result.wasSuccessful() or torch.cuda.is_initialized():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
