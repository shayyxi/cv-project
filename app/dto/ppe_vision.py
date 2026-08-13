from typing import List

from pydantic import BaseModel, Field


class BoundingBoxDTO(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class PPEDetectionDTO(BaseModel):
    label: str
    confidence: float

    # Original-image coordinates
    box: BoundingBoxDTO

    # Person-crop coordinates
    crop_box: BoundingBoxDTO


class ComplianceDTO(BaseModel):
    helmet: bool
    vest: bool
    boots: bool
    compliant: bool


class VisionDetectionDTO(BaseModel):
    """
    Represents one detected person.
    """

    label: str
    confidence: float
    box: BoundingBoxDTO
    is_sensitive: bool

    person_id: int

    compliance: ComplianceDTO

    ppe: List[PPEDetectionDTO] = Field(
        default_factory=list
    )


class VisionResultDTO(BaseModel):
    worker_count: int

    detections: List[VisionDetectionDTO] = Field(
        default_factory=list
    )