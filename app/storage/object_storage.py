from abc import ABC, abstractmethod
from pathlib import Path


ObjectPath = Path | str


class ObjectStorage(ABC):
    @abstractmethod
    def save_raw_image(
        self,
        camera_id: str,
        image_bytes: bytes,
    ) -> ObjectPath:
        raise NotImplementedError

    @abstractmethod
    def save_processed_image(
        self,
        camera_id: str,
        image_bytes: bytes,
    ) -> ObjectPath:
        raise NotImplementedError

    @abstractmethod
    def load_image(
            self,
            path: ObjectPath,
    ) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def delete(self, path: ObjectPath) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: ObjectPath) -> bool:
        raise NotImplementedError