"""
Render captures through the vision pipeline without the database.

    python scripts/preview_image.py data/raw/12846/12846_2026-09-11T13-30-32Z.jpg
    python scripts/preview_image.py data/raw/12846
    python scripts/preview_image.py data/raw --limit 5

Runs crop -> PPEVisionEngine -> translate -> face blur -> render, exactly
as the processing service does, but writes nothing to the database and
delivers nothing. Annotated JPEGs go to data/processed2/<camera_id>/<name>
and one row per image is appended to data/processed2/summary.csv.

Given a folder, every image under it (recursively) is processed. Images
whose output already exists are skipped, so an interrupted run can be
resumed with the same command. --force re-renders them. --limit N stops
after N newly rendered images.

The camera id is taken from the file name prefix (before the first
underscore) unless --camera-id is given.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from app.processing.cv.ppe_vision_engine import PPEVisionEngine
from app.processing.cv.vision_renderer import VisionRenderer
from app.processing.privacy.face_blur_service import FaceBlurPrivacyService
from app.processing.Image_cropper import ImageCropper

REPO_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_DIR = REPO_ROOT / "data" / "processed2"
SUMMARY_PATH = PREVIEW_DIR / "summary.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SUMMARY_COLUMNS = [
    "file",
    "camera_id",
    "seconds",
    "workers",
    "compliant",
    "non_compliant",
]


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def camera_id_for(image_path: Path, override: str | None) -> str:
    return override or image_path.name.split("_", 1)[0]


def append_summary(row: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_PATH.exists()

    with SUMMARY_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def render_one(
    image_path: Path,
    camera_id: str,
    out_path: Path,
    cropper: ImageCropper,
    engine: PPEVisionEngine,
    privacy: FaceBlurPrivacyService,
    renderer: VisionRenderer,
    verbose: bool,
) -> dict:
    image_bytes = image_path.read_bytes()

    cropped, region = cropper.crop(image_bytes, camera_id)

    started = time.perf_counter()
    result = engine.process_image(cropped)
    elapsed = time.perf_counter() - started

    result = cropper.translate_result_to_original(result, region)
    blurred = privacy.apply_privacy_blur(image_bytes, result)
    annotated = renderer.draw_original(blurred, result)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(annotated)

    compliant = sum(1 for p in result.detections if p.compliance.compliant)

    if verbose:
        for person in result.detections:
            polygons = len(person.mask) if person.mask else 0
            ppe = ", ".join(d.label for d in person.ppe) or "-"
            status = "COMPLIANT" if person.compliance.compliant else "NON-COMPLIANT"
            print(
                f"  P{person.person_id} conf={person.confidence:.2f} "
                f"{status} ppe=[{ppe}] mask_polygons={polygons}"
            )

    return {
        "file": str(image_path),
        "camera_id": camera_id,
        "seconds": f"{elapsed:.1f}",
        "workers": result.worker_count,
        "compliant": compliant,
        "non_compliant": result.worker_count - compliant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", type=Path, help="image file or folder")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--out", type=Path, default=None, help="single-file output path")
    parser.add_argument("--limit", type=int, default=None, help="stop after N new renders")
    parser.add_argument("--force", action="store_true", help="re-render existing outputs")
    args = parser.parse_args()

    path: Path = args.path
    if not path.exists():
        print(f"Not found: {path}")
        return 1

    images = collect_images(path)
    if not images:
        print(f"No images under {path}")
        return 1

    single = path.is_file()

    if args.out is not None and not single:
        print("--out only applies to a single image")
        return 1

    cropper = ImageCropper()
    engine = PPEVisionEngine()
    privacy = FaceBlurPrivacyService()
    renderer = VisionRenderer()

    print(f"{len(images)} image(s), device={engine._device}, output={PREVIEW_DIR}")

    rendered = skipped = failed = 0
    batch_started = time.perf_counter()

    for index, image_path in enumerate(images, start=1):
        camera_id = camera_id_for(image_path, args.camera_id)
        out_path = args.out or PREVIEW_DIR / camera_id / image_path.name

        if out_path.exists() and not args.force:
            skipped += 1
            print(f"[{index}/{len(images)}] {image_path.name} skipped (exists)")
            continue

        try:
            row = render_one(
                image_path,
                camera_id,
                out_path,
                cropper,
                engine,
                privacy,
                renderer,
                verbose=single,
            )
        except Exception as exc:  # keep the batch going
            failed += 1
            print(f"[{index}/{len(images)}] {image_path.name} FAILED: {exc}")
            continue

        rendered += 1
        append_summary(row)

        print(
            f"[{index}/{len(images)}] {image_path.name} {row['seconds']}s "
            f"workers={row['workers']} compliant={row['compliant']} -> {out_path}"
        )

        if args.limit is not None and rendered >= args.limit:
            print(f"Reached --limit {args.limit}")
            break

    total = time.perf_counter() - batch_started
    print(
        f"Done: rendered={rendered} skipped={skipped} failed={failed} "
        f"in {total / 60:.1f} min. Summary: {SUMMARY_PATH}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
