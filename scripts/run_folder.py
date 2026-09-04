import argparse
from pathlib import Path

from app.application.bootstrap import Application
from app.config import settings
from app.utils.clock import utc_now
from app.utils.hashing import sha256_bytes
from app.utils.logging import configure_logging

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest all images from a local folder and process them once.",
    )
    parser.add_argument("folder", type=Path, help="Folder containing images")
    parser.add_argument(
        "--camera-id",
        default="local",
        help="Camera id to tag the images with (default: local)",
    )
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()

    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    configure_logging()
    app = Application()

    raw_dir = settings.local_raw_dir / args.camera_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    ingested = 0

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_bytes = file.read_bytes()

        raw_path = raw_dir / file.name
        raw_path.write_bytes(image_bytes)

        app.image_job_repository.create_downloaded(
            camera_id=args.camera_id,
            remote_url=file.as_uri(),
            raw_image_path=str(raw_path),
            sha256=sha256_bytes(image_bytes),
            downloaded_at=utc_now(),
        )
        ingested += 1

    print(f"Ingested {ingested} images from {folder}")

    processed = 0

    while app.processing_service.process_next():
        processed += 1

    if processed:
        app.analytics_repository.refresh_views()

    print(f"Processed {processed} images.")


if __name__ == "__main__":
    main()
