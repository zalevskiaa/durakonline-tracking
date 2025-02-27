from ultralytics import YOLO
# import cv2
import numpy as np
import os


from torch_classifier import \
    mobilenetv3_rank_classifier, \
    mobilenetv3_suit_classifier, \
    mobilenetv3_take_classifier


model_detection_path = os.path.join(
    os.getcwd(), 'app/weights', 'cards-detection-train12-best.pt')
model_detection_players_path = os.path.join(
    os.getcwd(), 'app/weights', 'players-detection-train-best.pt')


class CardsDetector:
    def __init__(self):
        self.model_det = YOLO(model_detection_path)

        self.mobilenetv3_rank_classifier = mobilenetv3_rank_classifier
        self.mobilenetv3_suit_classifier = mobilenetv3_suit_classifier

        self.model_det_players = YOLO(model_detection_players_path)
        self.mobilenetv3_take_classifier = mobilenetv3_take_classifier

    @staticmethod
    def _carve_corner(image: np.ndarray, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        corner = image[y1:y2, x1:x2, :]
        return corner

    def predict(self, batch_images: list[np.ndarray]):
        if not batch_images:
            return {
                'batch_boxes': [],
                'batch_corners': [],
                'batch_ranks': [],
                'batch_suits': [],

                'batch_players_boxes': [],
                'batch_players_classes': [],
            }
        batch_results_det = self.model_det.predict(
            batch_images, conf=0.7, verbose=False
        )

        batch_boxes, batch_corners = [], []
        for image, result_det in zip(batch_images, batch_results_det):
            boxes = result_det.boxes.xyxy.detach().cpu().numpy()
            boxes = boxes.round().astype(int)

            corners = [
                CardsDetector._carve_corner(image, box) for box in boxes
            ]

            batch_boxes.append(boxes)
            batch_corners.append(corners)

        batch_ranks = []
        batch_suits = []

        for corners in batch_corners:
            ranks = self.mobilenetv3_rank_classifier.predict(corners)
            suits = self.mobilenetv3_suit_classifier.predict(corners)

            batch_ranks.append(ranks)
            batch_suits.append(suits)

        # players detection
        batch_results_players_det = self.model_det_players.predict(
            batch_images, conf=0.7, verbose=False
        )
        batch_players_boxes, batch_players = [], []
        for image, result_det in zip(batch_images, batch_results_players_det):
            boxes = result_det.boxes.xyxy.detach().cpu().numpy()
            boxes = boxes.round().astype(int)

            players = [
                CardsDetector._carve_corner(image, box)
                for box in boxes
            ]

            batch_players_boxes.append(boxes)
            batch_players.append(players)

        # players classification
        batch_players_take = []
        for players in batch_players:
            players_take = self.mobilenetv3_take_classifier.predict(players)
            batch_players_take.append(players_take)

        return {
            'batch_boxes': batch_boxes,
            'batch_corners': batch_corners,
            'batch_ranks': batch_ranks,
            'batch_suits': batch_suits,

            'batch_players_boxes': batch_players_boxes,
            'batch_players_take': batch_players_take,
        }
