from app.dto import BoundingBoxDTO, VisionDetectionDTO, VisionResultDTO
from app.processing.cv.vision_engine import VisionEngine


class DummyVisionEngine(VisionEngine):
    def process_image(self, image_bytes: bytes) -> VisionResultDTO:
        return VisionResultDTO(
            worker_count=1,
            detections=[
                VisionDetectionDTO(
                    label="person",
                    confidence=0.90,
                    box=BoundingBoxDTO(
                        x_min=100,
                        y_min=100,
                        x_max=300,
                        y_max=500,
                    ),
                    is_sensitive=False,
                )
            ],
        )