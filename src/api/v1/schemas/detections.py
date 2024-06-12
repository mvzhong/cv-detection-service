from pydantic import BaseModel


class DetectionRequestBase64(BaseModel):
    image_base64: str


class DetectionRequestImage(BaseModel):
    image_data: str


class DetectedObject(BaseModel):
    label: str
    confidence: float
    area_ratio: float


class DetectionResponse(BaseModel):
    objects: list[DetectedObject]
