"""Create title, result, and caption graphics for the polished demo video."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 960, 720
BACKGROUND = "#101b15"
PANEL = "#1b2a21"
ACCENT = "#d6a64b"
WHITE = "#f3f4ed"
MUTED = "#aab5ab"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def centered(draw: ImageDraw.ImageDraw, text: str, y: int, size: int, colour: str, bold: bool = False) -> None:
    selected = font(size, bold)
    box = draw.textbbox((0, 0), text, font=selected)
    x = (WIDTH - (box[2] - box[0])) // 2
    draw.text((x, y), text, fill=colour, font=selected)


def card() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.ellipse((680, -220, 1120, 220), outline="#314936", width=2)
    draw.ellipse((-180, 500, 210, 890), outline="#243b2d", width=2)
    draw.line((180, 110, 780, 110), fill="#385042", width=2)
    draw.line((180, 610, 780, 610), fill="#385042", width=2)
    return image, draw


def title_card(path: Path) -> None:
    image, draw = card()
    centered(draw, "TIMBERSIGHT", 195, 58, ACCENT, bold=True)
    centered(draw, "REAL-TIME WOOD DEFECT DETECTION", 292, 29, WHITE, bold=True)
    centered(draw, "Camera  |  PyTorch model  |  Annotated result", 362, 22, MUTED)
    centered(draw, "LIVE KNOT", 478, 18, "#58d68d", bold=True)
    centered(draw, "DEAD KNOT", 518, 18, "#ff7b72", bold=True)
    image.save(path, optimize=True)


def end_card(path: Path) -> None:
    image, draw = card()
    centered(draw, "TIMBERSIGHT", 132, 48, ACCENT, bold=True)
    centered(draw, "90.4% mAP@50", 240, 44, WHITE, bold=True)
    centered(draw, "47.9 ms warmed local inference", 320, 29, WHITE, bold=True)
    centered(draw, "Faster R-CNN  |  MobileNetV3  |  PyTorch", 410, 23, MUTED)
    centered(draw, "REAL-TIME WOOD INSPECTION SYSTEM", 498, 19, ACCENT, bold=True)
    image.save(path, optimize=True)


def caption(path: Path, text: str) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    selected = font(22, bold=True)
    box = draw.textbbox((0, 0), text, font=selected)
    text_width = box[2] - box[0]
    x1 = (WIDTH - text_width) // 2 - 16
    x2 = (WIDTH + text_width) // 2 + 16
    draw.rounded_rectangle((x1, 98, x2, 142), radius=10, fill=(5, 12, 8, 210))
    draw.text((x1 + 16, 108), text, fill=WHITE, font=selected)
    image.save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace existing assets")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "demo_images/video_assets"
    outputs = {
        "title.png": lambda path: title_card(path),
        "end.png": lambda path: end_card(path),
        "caption_01.png": lambda path: caption(path, "Knot-free surface - no defect detected"),
        "caption_02.png": lambda path: caption(path, "Dead knot - localized and classified"),
        "caption_03.png": lambda path: caption(path, "Live knot - localized and classified"),
        "caption_04.png": lambda path: caption(path, "New camera angle - evaluated in milliseconds"),
    }
    for name, create in outputs.items():
        destination = output / name
        if destination.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        create(destination)


if __name__ == "__main__":
    main()
