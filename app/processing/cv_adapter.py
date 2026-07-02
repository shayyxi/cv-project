from pathlib import Path
from app.dto import VisionResultDTO

class VisionEngine:
    def process_image(self, image_path: Path) -> VisionResultDTO:
        raise NotImplementedError


class DummyVisionEngine(VisionEngine):
    def process_image(self, image_path: Path) -> VisionResultDTO:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return VisionResultDTO(
            worker_count=0,
            detections=[],
        )