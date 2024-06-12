from os import path
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.api.v1.schemas.detections import (
    DetectedObject,
    DetectionRequestBase64,
    DetectionResponse,
)
from src.services.detector.detector_impl import Detector

# TEST_VIDEO_PATH = path.abspath(path.join(path.dirname(__file__), "../../../test_videos/"))
MODEL_DATA_PATH = path.abspath("model_data/")


class DetectorService:
    def __init__(self):
        config_path = path.join(
            MODEL_DATA_PATH, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        )
        model_path = path.join(MODEL_DATA_PATH, "frozen_inference_graph.pb")
        class_labels_path = path.join(MODEL_DATA_PATH, "coco.names")

        self.detector = Detector(config_path, model_path, class_labels_path)

    def detect_from_base64(self, request: DetectionRequestBase64) -> DetectionResponse:
        detected_objects = self.detector.detect_objects_base64(request.image_base64)

        return DetectionResponse(
            objects=[
                DetectedObject(
                    label=obj.label,
                    confidence=obj.confidence,
                    area_ratio=obj.area_ratio,
                )
                for obj in detected_objects
            ]
        )

    def detect_from_image(self, imgFile: UploadFile):
        detected_objects = self.detector.detect_objects_file(imgFile)

        return DetectionResponse(
            objects=[
                DetectedObject(
                    label=obj.label,
                    confidence=obj.confidence,
                    area_ratio=obj.area_ratio,
                )
                for obj in detected_objects
            ]
        )

    def get_output(self) -> FileResponse:
        output_path = self.detector.get_output_file_path()

        image = Path(output_path)
        if not image.exists():
            raise HTTPException(404, "No output found")

        return FileResponse(output_path)
