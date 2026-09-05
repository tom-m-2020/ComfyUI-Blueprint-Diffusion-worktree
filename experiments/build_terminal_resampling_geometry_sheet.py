"""Build the Phase 28 selected/control comparison sheet."""
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent / "terminal_resampling_geometry_qualification_results"
cases = ("SQUARE_MULTI_OBJECT", "PORTRAIT_ASTRONAUT", "LANDSCAPE_BRIDGE")
rows = []
for name in cases:
    images = []
    for stem in ("selected_0", "ordinary_tiled"):
        image = Image.open(root / name / f"{stem}.png").convert("RGB")
        image.thumbnail((900, 650), Image.Resampling.LANCZOS)
        images.append(image.copy())
    width = sum(image.width for image in images) + 20
    height = max(image.height for image in images) + 40
    row = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(row)
    x = 0
    for label, image in zip(("Blueprint terminal resampling", "ordinary tiled-local"), images):
        row.paste(image, (x, 30))
        draw.text((x + 4, 6), f"{name}: {label}", fill="black")
        x += image.width + 20
    rows.append(row)
sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
y = 0
for row in rows:
    sheet.paste(row, (0, y))
    y += row.height
sheet.save(root / "PHASE28_COMPARISON.jpg", quality=92)
