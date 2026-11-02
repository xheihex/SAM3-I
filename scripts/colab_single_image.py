#!/usr/bin/env python3
"""Single-image SAM3-I inference for Colab; upstream files remain untouched."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sam3"))

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
import torch
import torch.nn.functional as F

from sam3 import build_sam3_image_model
from sam3.model import box_ops
from sam3.model.sam3_image_processor import Sam3Processor

DEFAULT_CHECKPOINT = Path("/home/kmg/models/sam3-i/checkpoint.pt")
DEFAULT_BPE = ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
ADAPTER_CONFIG = {"adapter_dim": 64, "adapter_heads": 4, "adapter_scale": 1.0}
TEXT_TYPES = {"concept": 1, "simple": 2, "complex": 4}


def validate_request(image_path, prompt, category, threshold):
    if not Path(image_path).is_file():
        raise FileNotFoundError(image_path)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a nonempty string.")
    if category not in TEXT_TYPES:
        raise ValueError(f"Unknown category: {category}")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1].")


def open_rgb(path):
    # Match the author's RGB/PIL path; preserve stored pixel orientation and size.
    with Image.open(path) as image:
        return image.convert("RGB")


class CpuMaskProcessor(Sam3Processor):
    """Use upstream input/routing/forward; resize selected masks one by one on CPU.

    Deliberately omits the unused, large masks_logits probability tensor.
    Interpolation is float32, so low precision GPU boundary pixels may differ.
    """

    @torch.inference_mode()
    def _forward_grounding(self, state):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
            text_type=state["text_type"],
        )
        probs = (
            outputs["pred_logits"].sigmoid()
            * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        ).squeeze(-1)
        keep = probs > self.confidence_threshold
        scores = probs[keep].to(device="cpu", dtype=torch.float32)
        boxes = outputs["pred_boxes"][keep].to(device="cpu", dtype=torch.float32)
        low_res_masks = outputs["pred_masks"][keep].to(device="cpu", dtype=torch.float32)
        if not (torch.isfinite(scores).all() and torch.isfinite(boxes).all()
                and torch.isfinite(low_res_masks).all() and torch.isfinite(probs).all()):
            raise RuntimeError("Non-finite prediction; try --precision fp32 or a BF16 GPU.")
        height, width = state["original_height"], state["original_width"]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes *= torch.tensor([width, height, width, height], dtype=torch.float32)
        masks = torch.empty((len(scores), 1, height, width), dtype=torch.bool)
        for index in range(len(scores)):
            probability = F.interpolate(
                low_res_masks[index:index + 1, None],
                size=(height, width), mode="bilinear", align_corners=False,
            ).sigmoid()
            masks[index] = probability[0] > 0.5
        state.pop("masks_logits", None)
        state.update(masks=masks, boxes=boxes, scores=scores)
        return state


def choose_precision(requested, native_bf16):
    if requested == "auto":
        return "bf16" if native_bf16 else "fp16"
    if requested == "bf16" and not native_bf16:
        raise ValueError("Native BF16 is unavailable; use auto, fp16, or fp32.")
    return requested


def predict(model, image, prompt, category, threshold, *, device, precision):
    processor = CpuMaskProcessor(
        model, device=device, resolution=1008, confidence_threshold=threshold,
    )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    amp = torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()
    with torch.inference_mode(), amp:
        state = processor.set_image(image)
        state = processor.set_text_prompt(
            prompt=prompt, text_type=TEXT_TYPES[category], state=state,
        )
    return state


def save_results(image, state, output_dir, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = state["masks"].cpu().numpy()
    scores = state["scores"].float().cpu().tolist()
    boxes = state["boxes"].float().cpu().tolist()
    width, height = image.size
    if masks.shape != (len(scores), 1, height, width) or len(boxes) != len(scores):
        raise ValueError("Prediction shapes do not match the input image.")
    union = np.zeros((height, width), dtype=np.bool_)
    preview = image.convert("RGBA")
    predictions = []
    # Stable order by score, without dropping any threshold-passing instance.
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    palette = [(255, 76, 76), (38, 166, 255), (255, 194, 38), (85, 209, 108)]
    for rank, index in enumerate(order):
        mask = masks[index, 0]
        union |= mask
        filename = f"mask_{rank:03d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / filename)
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        predictions.append({
            "score": scores[index], "box_xyxy": boxes[index],
            "area_pixels": int(mask.sum()), "mask_file": filename,
            "rle": {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")},
        })
        color = palette[rank % len(palette)]
        layer = np.zeros((height, width, 4), dtype=np.uint8)
        layer[mask, :3] = color
        layer[mask, 3] = 90
        preview = Image.alpha_composite(preview, Image.fromarray(layer))
    draw = ImageDraw.Draw(preview)
    for rank, index in enumerate(order):
        x0, y0, x1, y1 = boxes[index]
        # Preserve raw boxes in JSON; clip only the preview coordinates.
        coords = [max(0, min(width - 1, x0)), max(0, min(height - 1, y0)),
                  max(0, min(width - 1, x1)), max(0, min(height - 1, y1))]
        draw.rectangle(coords, outline=palette[rank % len(palette)], width=2)
        draw.text((coords[0], coords[1]), f"{rank}: {scores[index]:.3f}", fill="white")
    preview.convert("RGB").save(output_dir / "overlay.png")
    Image.fromarray(union.astype(np.uint8) * 255).save(output_dir / "union_mask.png")
    cutout = image.convert("RGBA")
    cutout.putalpha(Image.fromarray(union.astype(np.uint8) * 255))
    cutout.save(output_dir / "cutout.png")
    result = {
        **metadata, "width": width, "height": height,
        "num_instances": len(predictions), "predictions": predictions,
        "postprocess": "CPU float32 bilinear; sigmoid > 0.5; no NMS or top-1",
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def mmap_checkpoint_state(checkpoint_path):
    """Same model tensor mapping as the author, with reclaimable file-backed pages.

    Use a trusted checkpoint on a local disk; torch.load preserves the author's
    weights_only=False setting for training-checkpoint metadata compatibility.
    """
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False,
    )
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint state dict or a model state dict.")
    return checkpoint


def build_model(checkpoint_path, bpe_path):
    # Scope the I/O substitution to construction; all matching/loading remains upstream.
    with patch("sam3.model_builder._load_raw_checkpoint", mmap_checkpoint_state):
        return build_sam3_image_model(
            checkpoint_path=str(Path(checkpoint_path).resolve()),
            bpe_path=str(Path(bpe_path).resolve()), load_from_HF=False,
            device="cuda", eval_mode=True, inst_stage="3",
            enable_inst_interactivity=False, compile=False, adapter_config=ADAPTER_CONFIG,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--category", choices=TEXT_TYPES, default="simple")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bpe", type=Path, default=DEFAULT_BPE)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Use a new directory for each query.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    args = parser.parse_args(argv)
    validate_request(args.image, args.prompt, args.category, args.threshold)
    for path in (args.checkpoint, args.bpe):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; select a new directory for this query.")
    if not torch.cuda.is_available():
        parser.error("Real inference requires a CUDA GPU. Use the offline validator locally.")
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    precision = choose_precision(args.precision, native_bf16)
    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(json.dumps({
        "gpu": props.name, "total_gib": total_bytes / 2**30,
        "free_gib": free_bytes / 2**30, "precision": precision,
        "torch": torch.__version__,
    }, indent=2), flush=True)
    # Read image before constructing/loading the large model.
    image = open_rgb(args.image)
    started = time.perf_counter()
    try:
        model = build_model(args.checkpoint, args.bpe)
        state = predict(
            model, image, args.prompt, args.category, args.threshold,
            device="cuda", precision=precision,
        )
        result = save_results(image, state, args.output_dir, {
            "image_file": str(args.image), "prompt": args.prompt,
            "category": args.category, "text_type": TEXT_TYPES[args.category],
            "threshold": args.threshold, "precision": precision,
            "checkpoint": str(args.checkpoint), "inst_stage": "3",
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        })
    except torch.OutOfMemoryError as error:
        raise RuntimeError(
            "GPU memory exhausted. Batch size is already 1; select a GPU with more "
            "free VRAM. Original image masks are already resized on CPU."
        ) from error
    print(f"Saved {result['num_instances']} instances to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
