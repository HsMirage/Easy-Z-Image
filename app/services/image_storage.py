from pathlib import Path

from PIL import Image, ImageOps

from app.config import settings

THUMBNAIL_DIR = "_thumbs"
THUMBNAIL_SIZE = 384
THUMBNAIL_QUALITY = 82


def build_image_url(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    return f"/storage/{relative_path}"


def build_thumbnail_path(image_path: str | None) -> str | None:
    if not image_path:
        return None
    return str(Path(THUMBNAIL_DIR) / Path(image_path).with_suffix(".webp"))


def build_thumbnail_url(image_path: str | None) -> str | None:
    thumbnail_path = build_thumbnail_path(image_path)
    if not thumbnail_path:
        return None

    thumbnail_file = settings.STORAGE_ROOT / thumbnail_path
    if not thumbnail_file.exists():
        return None

    return build_image_url(thumbnail_path)


def ensure_thumbnail(image_path: str | None) -> str | None:
    if not image_path:
        return None

    source_file = settings.STORAGE_ROOT / image_path
    if not source_file.exists():
        return None

    thumbnail_path = build_thumbnail_path(image_path)
    if not thumbnail_path:
        return None

    thumbnail_file = settings.STORAGE_ROOT / thumbnail_path
    if thumbnail_file.exists():
        return thumbnail_path

    thumbnail_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(source_file) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
            img.save(
                thumbnail_file,
                format="WEBP",
                quality=THUMBNAIL_QUALITY,
                method=6,
            )
    except Exception as exc:
        thumbnail_file.unlink(missing_ok=True)
        print(f"[ImageStorage] Failed to create thumbnail for {image_path}: {exc}")
        return None

    return thumbnail_path
