# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import os
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger("dinov2")


class HemaStandardDataset(VisionDataset):
    def __init__(
        self,
        *,
        root: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.patches = []

        all_dataset_files = Path(root).glob("*.txt")

        for dataset_file in all_dataset_files:
            print("Loading ", dataset_file)
            with open(dataset_file, "r") as file:
                content = file.read()
            file_list = content.splitlines()
            self.patches.extend(file_list)
        self.true_len = len(self.patches)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        try:
            image, filepath = self.get_image_data(index)
        except Exception as e:
            adjusted_index = index % self.true_len
            filepath = self.patches[adjusted_index]
            print(f"can not read image for sample {index, e,filepath}")
            return self.__getitem__(index + 1)

        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, filepath

    def get_image_data(self, index: int, dimension=224) -> Image:
        # Load image from jpeg file
        print(self.true_len)
        print(index)
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        patch = Image.open(filepath).convert(mode="RGB").resize((dimension, dimension), Image.Resampling.LANCZOS)
        return patch, filepath

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))

    def __len__(self) -> int:
        return 120000000  # large number for infinite data sampling


class RBCDataset(VisionDataset):
    def __init__(
        self,
        *,
        root: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        aux_path: Optional[str] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.patches = []

        # walk across folders and get all images
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg") or name.endswith(".bmp"):
                    self.patches.append(os.path.join(path,name))
        leng_pre = len(self.patches)
        logger.info(f"Found {leng_pre} images in root folder")

        if aux_path is not None:
            logger.info(f"Adding aux path: {aux_path}")
            print("Adding aux path: ", aux_path)
            for path, subdirs, files in os.walk(aux_path):
                for name in files:
                    if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg") or name.endswith(".bmp"):
                        self.patches.append(os.path.join(path,name))
        leng_post = len(self.patches)
        logger.info(f"Found {leng_post} images in total")
                        
        self.true_len = len(self.patches)
        if shuffle:
            import random
            random.shuffle(self.patches)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        try:
            image, filepath = self.get_image_data(index)
        except Exception as e:
            adjusted_index = index % self.true_len
            filepath = self.patches[adjusted_index]
            print(f"can not read image for sample {index, e,filepath}")
            return self.__getitem__(index + 1)

        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, filepath

    def get_image_data(self, index: int, dimension=224) -> Image:
        # Load image from jpeg file
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        patch = Image.open(filepath).convert(mode="RGB").resize((dimension, dimension), Image.Resampling.LANCZOS)
        return patch, filepath

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))

    def __len__(self) -> int:
        return 120000000  # large number for infinite data sampling
    
    
class RBCDatasetAlbu(VisionDataset):
    def __init__(
        self,
        *,
        root: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        aux_path: Optional[str] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.patches = []

        # walk across folders and get all images
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg") or name.endswith(".bmp"):
                    self.patches.append(os.path.join(path,name))
        
        leng_pre = len(self.patches)
        logger.info(f"Found {leng_pre} images in root folder")
        if aux_path is not None:
            logger.info(f"Adding aux path: {aux_path}")
            print("Adding aux path: ", aux_path)
            for path, subdirs, files in os.walk(aux_path):
                for name in files:
                    if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg") or name.endswith(".bmp"):
                        self.patches.append(os.path.join(path,name))
        leng_post = len(self.patches)
        logger.info(f"Found {leng_post} images in total")
        
        self.true_len = len(self.patches)
        if shuffle:
            import random
            random.shuffle(self.patches)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        try:
            filepath = self.get_image_data(index)
        except Exception as e:
            adjusted_index = index % self.true_len
            filepath = self.patches[adjusted_index]
            print(f"can not read image for sample {index, e,filepath}")
            return self.__getitem__(index + 1)

        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(filepath, target)

        return image, target, filepath

    def get_image_data(self, index: int) -> str:
        # Load image from jpeg file
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        return  filepath

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))

    def __len__(self) -> int:
        return 120000000  # large number for infinite data sampling
