from app.dto import VisionResultDTO
from app.processing.privacy.privacy_service import PrivacyService


class DummyPrivacyService(PrivacyService):
    def apply_privacy_blur(
        self,
        image_bytes: bytes,
        vision_result: VisionResultDTO,
    ) -> bytes:
        return image_bytes