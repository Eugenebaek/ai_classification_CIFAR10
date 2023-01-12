import os
import numpy as np
import pickle
from PIL import Image
from typing import Any, Tuple

from torchvision.datasets import VisionDataset


class CIFAR10(VisionDataset):

    """
    Class for loading CIFAR10 dataset (https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).
    This is taken and modified from the Pytorch vision datasets repository.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.

        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.

        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    # ---------- class attributes ----------
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar.html"
    tar_filename = "cifar-10-python.tar.gz"
    train_list = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_list = ["test_batch"]
    meta = {"filename": "batches.meta", "key": "label_names"}

    # ---------- initialization ----------
    def __init__(
        self, root: str, train: bool = True, transform=None, target_transform=None
    ) -> None:
        # we are extending the class VisionDataset
        super().__init__(root, transform=transform, target_transform=target_transform)

        # If train is True, extract the training dataset. Else, extract the testing dataset
        self.train = train
        if self.train:
            download_list = self.train_list
        else:
            download_list = self.test_list

        self.data = []
        self.targets = []

        # iterate over each file in download list and extract data into self.data and self.targets
        for file in download_list:
            file_path = os.path.join(self.root, self.base_folder, file)
            with open(file_path, "rb") as fp:
                entry = pickle.load(fp, encoding="latin1")
                self.data.append(entry["data"])
                # "Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs)"
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        # recieve data as numpy array with shape n, 3, 32, 32 for n images, 32x32 pixel value, and 3 color channels
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # transpose the data to be in form (# of images, height, width, color)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    # ---------- metadata ----------
    def _load_meta(self) -> None:
        """load meaningful names to the numeric labels"""
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as fp:
            data = pickle.load(fp, encoding="latin1")
            self.classes = data[self.meta["key"]]

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    # ---------- length ----------
    def __len__(self) -> int:
        """return length of dataset"""
        return len(self.data)

    # ---------- get item ----------
    def __getitem__(self, index) -> Tuple[Any, Any]:
        """returns the image and image label as a Tuple"""
        img = self.data[index]
        target = self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
