import torch
import torchvision
from torchvision.transforms import v2

import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


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


class YoloClsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 train: bool,
                 transform,
                 load_only_classes=False
                 ):
        self.root_path = root_path
        self.train = train
        self.transform = transform

        subset = 'train' if train else 'val'
        subset_path = os.path.join(root_path, subset)

        unique_labels = os.listdir(subset_path)
        self.unique_labels = sorted(unique_labels)

        self.images = []
        self.labels = []

        if not load_only_classes:
            for label in unique_labels:
                class_path = os.path.join(subset_path, label)
                class_filenames = os.listdir(class_path)
                for filename in class_filenames:
                    filepath = os.path.join(class_path, filename)
                    image = torchvision.io.read_image(filepath)

                    self.images.append(image)
                    self.labels.append(label)
            assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        label = self.labels[i]

        image = self.transform(image)
        label = self.unique_labels.index(label)

        return image, label


def make_run_folder(runs_foldername='runs'):
    runs_folderpath = os.path.join(os.getcwd(), runs_foldername)
    if not os.path.exists(runs_folderpath):
        os.mkdir(runs_folderpath)
    runs_list = os.listdir(runs_folderpath)

    if '.ipynb_checkpoints' in runs_list:
        runs_list.remove('.ipynb_checkpoints')

    if len(runs_list) > 0:
        runs_list = list(map(int, runs_list))
        current_run = max(runs_list) + 1
        run_foldername = f'{current_run:02}'
    else:
        current_run = 1
        run_foldername = f'{current_run:02}'
    run_folderpath = os.path.join(runs_folderpath, run_foldername)
    os.mkdir(run_folderpath)
    return run_folderpath


def make_transforms():
    train_transform = v2.Compose([
        ResizeByLarger(96*2),

        v2.RandomZoomOut(255, (1, 1.2), p=1),
        v2.RandomRotation((-15, 15), fill=255),

        SquarePad(255),

        v2.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.1, hue=0.05),

        v2.Resize(96),
        v2.ToDtype(torch.float32, scale=True),

        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
        ResizeByLarger(96),
        SquarePad(255),
        v2.ToDtype(torch.float32, scale=True),

        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def make_model(num_classes: int):
    model = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model.classifier[3] = torch.nn.Linear(1024, num_classes, bias=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model, loss_fn, optimizer, device


def do_train_epoch(model, loss_fn, optimizer, device, train_dataloader):
    model.train()

    losses = []
    correct, total = 0, 0

    for batch_images, batch_labels in tqdm(train_dataloader,
                                           total=len(train_dataloader),
                                           leave=False
                                           ):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        y_pred = model(batch_images)
        y_true = torch.nn.functional.one_hot(batch_labels,
                                             num_classes=y_pred.shape[1]
                                             ).float()

        assert y_pred.shape == y_true.shape, \
            f'y_pred.shape {y_pred.shape}, y_true.shape {y_true.shape}'

        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

        pred_labels = y_pred.argmax(1)
        correct += int(torch.count_nonzero(pred_labels == batch_labels))
        total += batch_labels.shape[0]

    mean_loss = torch.mean(torch.Tensor(losses))
    accuracy = correct / total
    return mean_loss, accuracy


def do_val_epoch(model, device, val_dataloader):
    model.eval()

    correct, total = 0, 0

    for batch_images, batch_labels in val_dataloader:
        with torch.no_grad():
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            y_pred = model(batch_images)
            pred_labels = y_pred.argmax(1)

            correct += int(torch.count_nonzero(pred_labels == batch_labels))
            total += batch_labels.shape[0]

    return correct / total


def do_train(train_parameters: dict):
    run_foldername = train_parameters['run_foldername']
    dataset_path = train_parameters['dataset_path']
    batch_size = train_parameters['batch_size']
    epochs = train_parameters['epochs']

    run_folderpath = make_run_folder(run_foldername)

    train_transform, val_transform = make_transforms()
    train_dataset = YoloClsDataset(
        dataset_path, train=True, transform=train_transform)
    val_dataset = YoloClsDataset(
        dataset_path, train=False, transform=val_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(train_dataset.unique_labels)

    model, loss_fn, optimizer, device = make_model(num_classes)
    print(f'device {device}')

    model = model.to(device)

    best_val_acc = None

    for epoch in range(epochs):
        print(f'epoch {epoch}')

        loss, train_acc = do_train_epoch(model, loss_fn,
                                         optimizer, device, train_dataloader
                                         )
        print(f'loss {loss:.4f}\ntrain_acc {train_acc:.4f}')

        val_acc = do_val_epoch(model, device, val_dataloader)
        print(f'val_acc {val_acc:.4f}')

        torch.save(model, os.path.join(run_folderpath, 'last.pt'))
        if best_val_acc is None or val_acc > best_val_acc:
            torch.save(model, os.path.join(run_folderpath, 'best.pt'))
            best_val_acc = val_acc

        print('---------------')

    return model
