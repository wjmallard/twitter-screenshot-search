"""Batch OCR runner for screenshot ingestion."""

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from tqdm import tqdm
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

import config
from db import get_conn, images_in_db, upsert_screenshot
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".tiff", ".bmp"}

# YYYYMMDD_HHMMSS--{original_name}--{shortuuid}.ext
FILENAME_RE = re.compile(r"^(\d{8}_\d{6})--")


def parse_created_at(filename: str) -> datetime | None:
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def preprocess(img: Image.Image) -> Image.Image:
    """Apply CLAHE contrast enhancement for dark-mode screenshot OCR."""
    arr = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(16, 16))
    return Image.fromarray(clahe.apply(arr))


def process_image(path: Path) -> dict:
    """Run OCR and extract metadata for a single image. Returns a dict for upsert."""
    img = Image.open(path)
    width, height = img.size
    ocr_text = pytesseract.image_to_string(preprocess(img))
    created_at = parse_created_at(path.name)
    return {
        "file_path": str(path),
        "ocr_text": ocr_text,
        "created_at": created_at,
        "width": width,
        "height": height,
    }


def images_on_disk(root: Path) -> set[str]:
    """Recursively find image files under root, returned as absolute path strings."""
    return {
        str(p) for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    }


def ingest(root: Path, workers: int = config.TESSERACT_WORKERS):
    on_disk = images_on_disk(root)
    print(f"Found {len(on_disk)} images under {root}")

    with get_conn() as conn:
        in_db = images_in_db(conn)

    to_process = sorted(on_disk - in_db)
    skipped = len(on_disk) - len(to_process)
    if skipped:
        print(f"Skipping {skipped} already-ingested files")
    if not to_process:
        print("Nothing to ingest.")
        return

    errors = 0
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_image, Path(p)): p for p in to_process}
        with get_conn() as conn:
            for future in tqdm(as_completed(futures), total=len(to_process), unit="img"):
                path = futures[future]
                try:
                    result = future.result()
                    upsert_screenshot(
                        conn,
                        result["file_path"],
                        result["ocr_text"],
                        result["created_at"],
                        result["width"],
                        result["height"],
                    )
                    done += 1
                    if done % config.COMMIT_BATCH_SIZE == 0:
                        conn.commit()
                except Exception as e:
                    errors += 1
                    tqdm.write(f"ERROR {path}: {e}", file=sys.stderr)

    print(f"\nDone ({errors} errors)" if errors else "\nDone")


def main():
    ingest(config.SCREENSHOT_DIR)


if __name__ == "__main__":
    main()
