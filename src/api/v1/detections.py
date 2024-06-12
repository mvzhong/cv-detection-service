from fastapi import APIRouter, Depends, UploadFile

from src.api.v1.schemas.detections import (
    DetectionRequestBase64,
    DetectionResponse,
)
from src.services.detector.detector_service import DetectorService

router = APIRouter(prefix="/detections", tags=["Detection API"])


@router.get("")
def get_detection():
    return {"message": "Detection GET"}


@router.post("/base64", response_model=DetectionResponse)
def post_detection_base64(
    request: DetectionRequestBase64,
    detector_service: DetectorService = Depends(DetectorService),
):
    return detector_service.detect_from_base64(request)


@router.post("/image", response_model=DetectionResponse)
def post_detection_image(
    file: UploadFile,
    detector_service: DetectorService = Depends(DetectorService),
):
    return detector_service.detect_from_image(file)


# TODO call the service instead of accessing file directly
@router.get("/output")
def get_detection_output(detector_service: DetectorService = Depends(DetectorService)):
    return detector_service.get_output()
