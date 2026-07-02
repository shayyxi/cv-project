from abc import ABC, abstractmethod

from app.dto import VisionResultDTO


class PrivacyService(ABC):
    @abstractmethod
    def apply_privacy_blur(
        self,
        image_bytes: bytes,
        vision_result: VisionResultDTO,
    ) -> bytes:
        raise NotImplementedError