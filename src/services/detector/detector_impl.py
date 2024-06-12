import base64
from os import path

import cv2
import numpy as np
from fastapi import UploadFile

CONFIDENCE_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2

OUTPUT_PATH = path.abspath(path.dirname(__file__) + "/output")

np.random.seed(10)  # keep color generation consistent


class DetectedObject:
    def __init__(self, label: str, confidence: float, area_ratio: float):
        self.label = label
        self.confidence = confidence
        self.area_ratio = area_ratio


class Detector:
    def __init__(self, config_path, model_path, class_labels_path):
        self.config_path = config_path
        self.model_path = model_path
        self.class_labels_path = class_labels_path

        self.net = cv2.dnn.DetectionModel(self.model_path, self.config_path)
        self.net.setInputSize(320, 320)
        self.net.setInputScale((1.0 / 127.5, 1.0 / 127.5))
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.__read_classes()

    def __read_classes(self):
        with open(self.class_labels_path, "r") as f:
            self.classes_list = f.read().splitlines()

        self.classes_list.insert(
            0, "__Background__"
        )  # apparently model predicts index 0 as background
        self.color_list = np.random.uniform(
            low=0, high=255, size=(len(self.classes_list), 3)
        )

        print(self.classes_list)

    def detect_objects_base64(self, img_base64: str) -> list[DetectedObject]:
        try:
            img_binary = base64.b64decode(img_base64)
            np_img = np.asarray(bytearray(img_binary), dtype=np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        except Exception:
            print("Error decoding image")
            raise Exception("Error decoding image")

        return self._detect_objects(image)

    def detect_objects_file(self, imgFile: UploadFile) -> list[DetectedObject]:
        try:
            contents = imgFile.file.read()
            np_img = np.frombuffer(contents, dtype=np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        except Exception:
            print("Error parsing image file")
            raise Exception("Error parsing image file")

        return self._detect_objects(image)

    def get_output_file_path(self) -> str:
        return path.abspath(path.join(OUTPUT_PATH, "output.png"))

    def _detect_objects(self, image: cv2.typing.MatLike) -> list[DetectedObject]:
        height, width, _ = image.shape
        image_area = height * width

        class_label_ids, confidences, bboxes = self.net.detect(
            image, confThreshold=CONFIDENCE_THRESHOLD
        )

        bboxes = list(bboxes)
        confidences = list(np.array(confidences).reshape(1, -1)[0])
        confidences = list(map(float, confidences))

        # Non-maxima suppression of bounding boxes to remove overlaps
        # Returns index of boxes that have overlap below nms_threshold
        filtered_bbox_indicies = cv2.dnn.NMSBoxes(
            bboxes,
            confidences,
            score_threshold=SCORE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
        )

        detected_objects: list[DetectedObject] = []

        if len(filtered_bbox_indicies) != 0:
            for index in filtered_bbox_indicies:
                bbox = bboxes[index]
                class_confidence = confidences[index]
                class_label_id = class_label_ids[index]
                class_label = self.classes_list[class_label_id]

                _, _, w, h = bbox
                bbox_area = w * h
                area_ratio = bbox_area * 100.0 / image_area

                self.__draw_box(
                    image,
                    bbox,
                    class_label,
                    class_label_id,
                    class_confidence,
                    area_ratio,
                )
                detected_objects.append(
                    DetectedObject(
                        label=class_label,
                        confidence=class_confidence,
                        area_ratio=area_ratio,
                    )
                )

        write_path = path.abspath(path.join(OUTPUT_PATH, "output.png"))
        cv2.imwrite(write_path, image)

        return detected_objects

    def __draw_box(
        self, image, bbox, class_label, class_label_id, class_confidence, area_ratio
    ):
        x, y, w, h = bbox
        height, width, _ = image.shape
        image_area = height * width

        class_color = [int(c) for c in self.color_list[class_label_id]]

        cv2.rectangle(image, (x, y), (x + w, y + h), color=class_color, thickness=1)

        bbox_area = w * h
        area_ratio = bbox_area * 100.0 / image_area

        display_text = "{} : {:.2f} : {:.2f}%".format(
            class_label, class_confidence, area_ratio
        )
        cv2.putText(
            image,
            display_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_PLAIN,
            5,
            class_color,
            2,
        )

        line_width = int(min((0.3 * w), (0.3 * h)))
        # top left
        cv2.line(image, (x, y), (x + line_width, y), class_color, 5)
        cv2.line(image, (x, y), (x, y + line_width), class_color, 5)

        # top right
        cv2.line(image, (x + w, y), (x + w - line_width, y), class_color, 5)
        cv2.line(image, (x + w, y), (x + w, y + line_width), class_color, 5)

        # bottom left
        cv2.line(image, (x, y + h), (x + line_width, y + h), class_color, 5)
        cv2.line(image, (x, y + h), (x, y + h - line_width), class_color, 5)

        # bottom right
        cv2.line(image, (x + w, y + h), (x + w - line_width, y + h), class_color, 5)
        cv2.line(image, (x + w, y + h), (x + w, y + h - line_width), class_color, 5)
