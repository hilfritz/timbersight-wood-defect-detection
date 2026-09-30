"""Create exact-size graphics for the second TimberSight demo edit."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 640
BACKGROUND = "#101b15"
ACCENT = "#d6a64b"
WHITE = "#f3f4ed"
MUTED = "#aab5ab"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def centered(draw: ImageDraw.ImageDraw, text: str, y: int, size: int, colour: str, bold: bool = False) -> None:
    selected = font(size, bold)
    box = draw.textbbox((0, 0), text, font=selected)
    draw.text(((WIDTH - (box[2] - box[0])) // 2, y), text, fill=colour, font=selected)


def card() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.ellipse((920, -230, 1350, 200), outline="#314936", width=2)
    draw.ellipse((-160, 440, 220, 820), outline="#243b2d", width=2)
    draw.line((230, 86, 970, 86), fill="#385042", width=2)
    draw.line((230, 554, 970, 554), fill="#385042", width=2)
    return image, draw


def title_card(path: Path) -> None:
    image, draw = card()
    centered(draw, "TIMBERSIGHT", 150, 58, ACCENT, bold=True)
    centered(draw, "REAL-TIME WOOD DEFECT DETECTION", 245, 30, WHITE, bold=True)
    centered(draw, "Continuous camera inspection", 310, 23, MUTED)
    centered(draw, "CREATED BY HILFRITZ CAMALLERE", 380, 23, ACCENT, bold=True)
    centered(draw, "LIVE KNOT   |   DEAD KNOT", 450, 19, WHITE, bold=True)
    image.save(path, optimize=True)


def end_card(path: Path) -> None:
    image, draw = card()
    centered(draw, "TIMBERSIGHT", 108, 48, ACCENT, bold=True)
    centered(draw, "REAL-TIME CAMERA INSPECTION", 190, 28, WHITE, bold=True)
    centered(draw, "90.4% mAP@50", 270, 42, WHITE, bold=True)
    centered(draw, "49.4 ms warmed local inference", 340, 28, WHITE, bold=True)
    centered(draw, "Faster R-CNN  |  MobileNetV3  |  PyTorch", 423, 22, MUTED)
    centered(draw, "REAL-TIME WOOD INSPECTION SYSTEM", 485, 18, ACCENT, bold=True)
    image.save(path, optimize=True)


def caption(path: Path, text: str) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    selected = font(22, bold=True)
    box = draw.textbbox((0, 0), text, font=selected)
    text_width = box[2] - box[0]
    x1 = (WIDTH - text_width) // 2 - 16
    x2 = (WIDTH + text_width) // 2 + 16
    draw.rounded_rectangle((x1, 70, x2, 114), radius=10, fill=(5, 12, 8, 215))
    draw.text((x1 + 16, 80), text, fill=WHITE, font=selected)
    image.save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace existing assets")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "demo_images/video_assets"
    outputs = {
        "demo2_title.png": title_card,
        "demo2_end.png": end_card,
        "demo2_caption_01.png": lambda path: caption(path, "Continuous camera inspection"),
        "demo2_caption_02.png": lambda path: caption(path, "Live knot detection updates in real time"),
        "demo2_caption_03.png": lambda path: caption(path, "Live and dead classifications update instantly"),
    }
    for name, create in outputs.items():
        destination = output / name
        if destination.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        create(destination)


if __name__ == "__main__":
    main()
