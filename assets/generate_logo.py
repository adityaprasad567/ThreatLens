"""Generate assets/logo.png (square icon) and assets/logo_wide.png (icon + wordmark)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
SS = 4  # supersampling for smooth edges


def icon(size: int) -> Image.Image:
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=(17, 26, 46, 255), outline=(34, 48, 77, 255), width=SS * 2)
    cx, cy, r = s * 0.46, s * 0.46, s * 0.26
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(91, 141, 239, 255), width=int(s * 0.07))      # lens ring
    d.line([cx + r * 0.72, cy + r * 0.72, s * 0.80, s * 0.80], fill=(91, 141, 239, 255), width=int(s * 0.08))  # handle
    pw, ph = s * 0.11, s * 0.13                                                                       # check mark
    d.line([cx - pw, cy, cx - pw * 0.2, cy + ph * 0.6, cx + pw * 1.1, cy - ph * 0.6], fill=(47, 208, 143, 255), width=int(s * 0.055), joint="curve")
    return img.resize((size, size), Image.LANCZOS)


def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


icon(256).save(OUT / "logo.png")
wide = Image.new("RGBA", (760, 160), (0, 0, 0, 0))
wide.paste(icon(128), (16, 16))
ImageDraw.Draw(wide).text((170, 36), "ThreatLens", font=font(72), fill=(231, 236, 246, 255))
wide.save(OUT / "logo_wide.png")
print("Logos written to", OUT)
