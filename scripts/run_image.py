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
        description="Ingest a single local image and process it once.",
    )
    parser.add_argument("image", type=Path, help="Path to one image file")
    parser.add_argument(
        "--camera-id",
        default="local",
        help="Camera id to tag the image with (default: local)",
    )
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()

    if not image_path.is_file():
        raise SystemExit(f"Not a file: {image_path}")

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise SystemExit(f"Not an image: {image_path}")

    configure_logging()
    app = Application()

    raw_dir = settings.local_raw_dir / args.camera_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = image_path.read_bytes()

    raw_path = raw_dir / image_path.name
    raw_path.write_bytes(image_bytes)

    app.image_job_repository.create_downloaded(
        camera_id=args.camera_id,
        remote_url=image_path.as_uri(),
        raw_image_path=str(raw_path),
        sha256=sha256_bytes(image_bytes),
        downloaded_at=utc_now(),
    )

    print(f"Ingested {image_path}")

    processed = 0

    while app.processing_service.process_next():
        processed += 1

    if processed:
        app.analytics_repository.refresh_views()

    print(f"Processed {processed} images.")


if __name__ == "__main__":
    main()
