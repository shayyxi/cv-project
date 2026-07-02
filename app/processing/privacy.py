from app.dto import VisionResultDTO

# For now this is a placeholder. Later, the OpenCV implementation will replace this and blur faces/sensitive boxes.
class PrivacyService:
    def apply_privacy_blur(
        self,
        image_bytes: bytes,
        vision_result: VisionResultDTO,
    ) -> bytes:
        return image_bytes