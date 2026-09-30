"""Add the creator credit to a copy of the TimberSight title card."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    assets = project / "demo_images/video_assets"
    source = assets / "title.png"
    destination = assets / "title_hilfritz.png"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 23
    )
    text = "CREATED BY HILFRITZ CAMALLERE"
    box = draw.textbbox((0, 0), text, font=font)
    x = (image.width - (box[2] - box[0])) // 2
    draw.text((x, 420), text, fill="#d6a64b", font=font)
    image.save(destination, optimize=True)


if __name__ == "__main__":
    main()

