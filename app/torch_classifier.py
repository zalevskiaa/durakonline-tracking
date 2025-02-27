import torch
from torchvision.transforms import v2

import os
import matplotlib.pyplot as plt
import numpy as np


class ResizeByLarger(torch.nn.Module):
    def __init__(self, size=96):
        self.size = size

    def __call__(self, img):
        c, h, w = img.shape
        assert c == 3

        r = self.size / max(h, w)
        img = v2.functional.resize(
            img,
            [(round(min(self.size, r * h))), round(min(self.size, r * w))]
        )

        return img


class SquarePad(torch.nn.Module):
    def __init__(self, fill):
        self.fill = fill

    def __call__(self, img):
        c, h, w = img.shape
        assert c == 3

        size = max(w, h)
        ptb, plr = size - h, size - w
        pt, pl = round(ptb // 2), round(plr // 2)
        pb, pr = ptb - pt, plr - pl

        img = v2.functional.pad(img, [pl, pt, pr, pb], self.fill, 'constant')
        assert img.shape == (c, size, size)

        return img


def imshow_torch(img):
    img = img.detach().cpu().numpy()
    img = img.transpose((1, 2, 0))

    plt.grid(False)
    plt.axes(None)
    plt.imshow(img)
    plt.show()


def imshow_torch_batch(batch):
    imshow_torch(torch.cat(batch, dim=-1))


class MobileNetV3Classifier:
    def __init__(self, model_path: str, classes_names: list[str]):
        """
        model_path - path to pt
        classes_names - list of classes names in same order as in model
        """
        self.model = torch.load(
            model_path,
            weights_only=False,
            map_location=torch.device('cpu')
        )
        self.model.eval()

        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classes_names = classes_names.copy()

        self.val_transform = v2.Compose([
            ResizeByLarger(96),
            SquarePad(255),
            v2.ToDtype(torch.float32, scale=True),
        ])

    @staticmethod
    def _from_numpy(np_image: np.ndarray):
        assert isinstance(np_image, np.ndarray)
        assert np_image.dtype == np.uint8
        assert np_image.max() > 2
        assert len(np_image.shape) == 3
        assert np_image.shape[2] == 3

        return torch.from_numpy(np_image.transpose(2, 0, 1))

    @staticmethod
    def _validate_batch(batch):
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.float32
        assert batch.max() < 1.00001
        assert batch.shape == (batch.shape[0], 3, *batch.shape[2:4])

    def predict(self, np_images: list[np.ndarray]) -> list[str]:
        """
        images: list of np.array images (h, w, c) uint8 [0..255]
        RETURNS: list of predicted classes names
        """
        if not np_images:
            return []

        images = map(MobileNetV3Classifier._from_numpy, np_images)
        images = map(self.val_transform, images)
        images = list(images)

        batch = torch.stack(images, dim=0)
        MobileNetV3Classifier._validate_batch(batch)

        with torch.no_grad():
            preds = self.model.to(self.device)(
                batch.to(self.device)
            ).detach().cpu().argmax(1)

        labels = list(map(self.classes_names.__getitem__, preds))

        return labels


mobilenetv3_rank_classifier_path = os.path.join(
    os.getcwd(), 'app/weights',
    'corner-classification-rank-03-best.pt'
)
mobilenetv3_suit_classifier_path = os.path.join(
    os.getcwd(), 'app/weights',
    'corner-classification-suit-01-best.pt'
)
mobilenetv3_take_classifier_path = os.path.join(
    os.getcwd(), 'app/weights',
    'player-classification-take-01-best.pt'
)

# NOTE: correct order of classes names is required!
classes_names_rank = ['10', '6', '7', '8', '9', 'A', 'J', 'K', 'Q']
classes_names_suit = ['c', 'd', 'h', 's']
classes_names_take = ['0', '1']

mobilenetv3_rank_classifier = MobileNetV3Classifier(
    mobilenetv3_rank_classifier_path,
    classes_names_rank
)
mobilenetv3_suit_classifier = MobileNetV3Classifier(
    mobilenetv3_suit_classifier_path,
    classes_names_suit
)
mobilenetv3_take_classifier = MobileNetV3Classifier(
    mobilenetv3_take_classifier_path,
    classes_names_take
)
