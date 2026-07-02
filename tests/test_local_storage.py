from pathlib import Path

from app.storage.local_storage import LocalStorage


def test_save_raw_image(tmp_path: Path) -> None:
    storage = LocalStorage(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        failed_root=tmp_path / "failed",
    )

    saved_path = storage.save_raw_image(
        camera_id="6168",
        image_bytes=b"image-bytes",
    )

    assert saved_path.exists()
    assert saved_path.read_bytes() == b"image-bytes"
    assert "6168" in saved_path.name


def test_save_processed_image(tmp_path: Path) -> None:
    storage = LocalStorage(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        failed_root=tmp_path / "failed",
    )

    saved_path = storage.save_processed_image(
        camera_id="6168",
        image_bytes=b"processed-image-bytes",
    )

    assert saved_path.exists()
    assert saved_path.read_bytes() == b"processed-image-bytes"
    assert "processed" in str(saved_path)


def test_delete_file(tmp_path: Path) -> None:
    storage = LocalStorage(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        failed_root=tmp_path / "failed",
    )

    saved_path = storage.save_raw_image("6168", b"image-bytes")

    assert storage.exists(saved_path) is True

    storage.delete(saved_path)

    assert storage.exists(saved_path) is False