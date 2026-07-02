from abc import ABC, abstractmethod

from app.dto import VisionResultDTO


class VisionEngine(ABC):
    @abstractmethod
    def process_image(self, image_bytes: bytes) -> VisionResultDTO:
        raise NotImplementedError