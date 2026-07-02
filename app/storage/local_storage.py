from pathlib import Path

from app.config import settings
from app.storage.object_storage import ObjectPath, ObjectStorage
from app.utils.clock import utc_now


class LocalStorage(ObjectStorage):
    def __init__(
        self,
        raw_root: Path | None = None,
        processed_root: Path | None = None,
        failed_root: Path | None = None,
    ) -> None:
        self._raw_root = raw_root or settings.local_raw_dir
        self._processed_root = processed_root or settings.local_processed_dir
        self._failed_root = failed_root or settings.local_failed_dir

    def ensure_directories(self) -> None:
        self._raw_root.mkdir(parents=True, exist_ok=True)
        self._processed_root.mkdir(parents=True, exist_ok=True)
        self._failed_root.mkdir(parents=True, exist_ok=True)

    def save_raw_image(self, camera_id: str, image_bytes: bytes) -> Path:
        return self._save_image(
            root=self._raw_root,
            camera_id=camera_id,
            image_bytes=image_bytes,
        )

    def save_processed_image(self, camera_id: str, image_bytes: bytes) -> Path:
        return self._save_image(
            root=self._processed_root,
            camera_id=camera_id,
            image_bytes=image_bytes,
        )

    def delete(self, path: ObjectPath) -> None:
        file_path = Path(path)

        if file_path.exists():
            file_path.unlink()

    def exists(self, path: ObjectPath) -> bool:
        return Path(path).exists()

    def _save_image(
        self,
        root: Path,
        camera_id: str,
        image_bytes: bytes,
    ) -> Path:
        camera_dir = root / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)

        file_path = camera_dir / self._generate_filename(camera_id)
        file_path.write_bytes(image_bytes)

        return file_path

    def _generate_filename(self, camera_id: str) -> str:
        timestamp = utc_now().strftime("%Y-%m-%dT%H-%M-%SZ")
        return f"{camera_id}_{timestamp}.jpg"

    def load_image(
            self,
            path: ObjectPath,
    ) -> bytes:
        return Path(path).read_bytes()