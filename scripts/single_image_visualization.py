"""Render numbered SAM3-I instances and a confidence panel with Pillow only.

The caller supplies masks in exactly the same order as predictions. Instance
identifiers are one-based and remain the same in the overlay and score panel.
"""
from __future__ import annotations

import colorsys
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def instance_color(instance_id: int) -> tuple[int, int, int]:
    """Return a stable, bright RGB color for a one-based instance identifier."""
    if not isinstance(instance_id, int) or instance_id < 1:
        raise ValueError("instance_id must be a positive integer")
    hue = ((instance_id - 1) * 0.618033988749895 + 0.02) % 1.0
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.78, 0.98))


def _font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _validate_predictions(predictions):
    identifiers = []
    for prediction in predictions:
        instance_id = prediction["instance_id"]
        instance_color(instance_id)
        identifiers.append(instance_id)
        score = float(prediction["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction scores must be finite and in [0, 1].")
        box = prediction["box_xyxy"]
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
            raise ValueError("Prediction box_xyxy must contain four finite coordinates.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Instance identifiers must be unique.")


def _label_origin(box, badge_size, occupied, image_size):
    x0, y0, x1, y1 = box
    badge_width, badge_height = badge_size
    width, height = image_size
    candidates = [(x0, y0), (x1 - badge_width, y0),
                  (x0, y1 - badge_height), (x1 - badge_width, y1 - badge_height)]
    def clipped(candidate):
        return (max(0, min(width - badge_width, round(candidate[0]))),
                max(0, min(height - badge_height, round(candidate[1]))))
    def overlap(origin):
        x, y = origin
        return sum(max(0, min(x + badge_width, right) - max(x, left))
                   * max(0, min(y + badge_height, bottom) - max(y, top))
                   for left, top, right, bottom in occupied)
    origin = min((clipped(candidate) for candidate in candidates), key=overlap)
    occupied.append((origin[0], origin[1], origin[0] + badge_width, origin[1] + badge_height))
    return origin


def render_overlay(image: Image.Image, masks: np.ndarray, predictions: list[dict]) -> Image.Image:
    """Return original-size RGB image with translucent masks, boxes and IDs.

    No confidence values are drawn over the photograph; they belong in the panel
    returned by render_result_figure.
    """
    _validate_predictions(predictions)
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    width, height = image.size
    if masks.shape != (len(predictions), height, width):
        raise ValueError("Masks must have shape [K,H,W] or [K,1,H,W] matching the image.")
    overlay = image.convert("RGB")
    for mask, prediction in zip(masks, predictions):
        alpha = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 88)
        overlay.paste(instance_color(prediction["instance_id"]), (0, 0, width, height), alpha)

    draw = ImageDraw.Draw(overlay)
    scale = width / 1200
    line_width = max(2, round(3 * scale))
    font_size = max(13, round(24 * scale))
    font = _font(font_size, bold=True)
    pad = max(3, round(5 * scale))
    occupied = []
    for prediction in predictions:
        color = instance_color(prediction["instance_id"])
        raw_box = [float(value) for value in prediction["box_xyxy"]]
        x0, x1 = sorted((max(0, min(width - 1, raw_box[0])),
                         max(0, min(width - 1, raw_box[2]))))
        y0, y1 = sorted((max(0, min(height - 1, raw_box[1])),
                         max(0, min(height - 1, raw_box[3]))))
        box = (x0, y0, x1, y1)
        draw.rectangle(box, outline=color, width=line_width)
        label = str(prediction["instance_id"])
        text_box = draw.textbbox((0, 0), label, font=font)
        badge_width = min(width, text_box[2] - text_box[0] + 2 * pad)
        badge_height = min(height, text_box[3] - text_box[1] + 2 * pad)
        x, y = _label_origin(box, (badge_width, badge_height), occupied, image.size)
        draw.rounded_rectangle((x, y, x + badge_width - 1, y + badge_height - 1),
                               radius=max(2, round(3 * scale)), fill=color,
                               outline=(255, 255, 255), width=max(1, round(scale)))
        # Dark text stays readable on the bright instance-color badge.
        draw.text((x + pad - text_box[0], y + pad - text_box[1]),
                  label, font=font, fill=(15, 23, 42))
    return overlay


def render_result_figure(
    overlay: Image.Image,
    predictions: list[dict],
    *,
    max_width: int = 1200,
) -> Image.Image:
    """Return a display-size RGB figure with the confidence panel below the image.

    The original-size overlay can be saved separately. This combined artifact
    uses at most max_width pixels across so its labels remain readable when
    displayed in Colab, including a score table with many instances.
    """
    _validate_predictions(predictions)
    if max_width < 160:
        raise ValueError("max_width must be at least 160 pixels.")
    width = min(max_width, max(320, overlay.width))
    height = max(1, round(overlay.height * width / overlay.width))
    photo = overlay.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    font_size = max(12, min(18, round(width / 60)))
    body_font = _font(font_size)
    title_font = _font(font_size + 2, bold=True)
    pad = max(10, round(width / 60))
    title_height = font_size + 2 * pad
    columns = max(1, min(4, width // 270))
    row_height = font_size + 18
    rows = max(1, math.ceil(len(predictions) / columns))
    panel_height = title_height + rows * row_height + pad
    figure = Image.new("RGB", (width, height + panel_height), (248, 250, 252))
    figure.paste(photo, (0, 0))
    draw = ImageDraw.Draw(figure)
    draw.line((0, height, width, height), fill=(203, 213, 225), width=1)
    draw.text((pad, height + pad), "Instance confidence", fill=(15, 23, 42), font=title_font)
    if not predictions:
        draw.text((pad, height + title_height + 4), "No instances above the threshold.",
                  fill=(71, 85, 105), font=body_font)
        return figure
    column_width = (width - 2 * pad) / columns
    swatch_size = max(10, font_size - 2)
    for index, prediction in enumerate(predictions):
        row, column = divmod(index, columns)
        x = round(pad + column * column_width)
        y = height + title_height + row * row_height + 4
        color = instance_color(prediction["instance_id"])
        draw.rounded_rectangle((x, y + 3, x + swatch_size, y + 3 + swatch_size),
                               radius=2, fill=color)
        text = f"#{prediction['instance_id']}  {float(prediction['score']):.4f}"
        draw.text((x + swatch_size + 8, y), text, fill=(30, 41, 59), font=body_font)
    return figure
