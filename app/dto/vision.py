from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBoxDTO:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class VisionDetectionDTO:
    label: str
    confidence: float
    box: BoundingBoxDTO
    is_sensitive: bool = False


@dataclass(frozen=True)
class VisionResultDTO:
    worker_count: int
    detections: list[VisionDetectionDTO]