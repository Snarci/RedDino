# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import random

import numpy as np
from torchvision import transforms

from .transforms import GaussianBlur, make_normalize_transform
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
#import random
import random


logger = logging.getLogger("dinov2")


class DataAugmentationHEMA(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                # transforms.CenterCrop(global_crops_size),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # additional histopathology-specific augmentations
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                # transforms.CenterCrop(global_crops_size),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # additional histopathology-specific augmentations
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            ]
        )

        # auto-rand augmentations
        strong_aug = transforms.Compose(
            [
                v2.RandAugment()
               # v2.AutoAugment()
            ]
        )
        logger.info("Using auto-rand augmentations")

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([strong_aug, self.normalize])
        self.global_transfo2 = transforms.Compose([strong_aug, self.normalize])
        self.local_transfo = transforms.Compose([strong_aug, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class DataAugmentationDINO(object):
    def __init__(
    self,
    global_crops_scale,
    local_crops_scale,
    local_crops_number,
    global_crops_size=224,
    local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                # transforms.CenterCrop(global_crops_size),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # additional histopathology-specific augmentations
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                # transforms.CenterCrop(global_crops_size),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # additional histopathology-specific augmentations
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                #        transforms.RandomApply(
                #            [HEDJitter(factor=0.07)],
                #            p=0.5,
                #        ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,  # original p=0.8
                ),
                # additional histopathology-specific augmentations (don't use grayscale)
                transforms.RandomGrayscale(p=0.2),  # original p=0.2, changed
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)  # original p=1., changed

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),  # changed
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )
        self.resize_gc = transforms.Compose(
        [
            transforms.Resize(global_crops_size, interpolation=transforms.InterpolationMode.BICUBIC),
        ]
        )
        
        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class DataAugmentationAlbumentations(object):
    def __init__(
        self,
        cfg,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.cfg = cfg
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.mean = IMAGENET_DEFAULT_MEAN
        self.std = IMAGENET_DEFAULT_STD
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"cfg: {cfg}")
        logger.info("###################################")
        p = 0.85
        self.number_of_color_aug = cfg["number_of_color_aug"]
        self.number_of_distortion_aug = cfg["number_of_distortion_aug"]

        self.use_3rd_global_crop = cfg["use_3rd_global_crop"]
        self.cropping_global = A.Compose(
            [
                A.RandomResizedCrop(
                    global_crops_size, global_crops_size, scale=global_crops_scale, p=1.0
                )
            ]
        )
        

        self.cropping_local = A.Compose(
            [
                A.RandomResizedCrop(
                    local_crops_size, local_crops_size, scale=local_crops_scale, p=1.0
                )
            ]
        )
        
        self.color_augmentations = [
            A.Blur(p=p) if cfg["Blur"] else None,
            A.CLAHE(p=p) if cfg["CLAHE"] else None,
            A.ChannelShuffle(p=p) if cfg["ChannelShuffle"] else None,
            A.ChannelDropout(p=p) if cfg["ChannelDropout"] else None,
            A.ColorJitter(p=p) if cfg["ColorJitter"] else None,
            A.Defocus(p=p) if cfg["Defocus"] else None,
            A.Downscale(p=p) if cfg["Downscale"] else None,
            A.Emboss(p=p) if cfg["Emboss"] else None,
            A.Equalize(p=p) if cfg["Equalize"] else None,
            A.FancyPCA(p=p) if cfg["FancyPCA"] else None,
            A.GaussNoise(p=p) if cfg["GaussNoise"] else None,
            A.GaussianBlur(p=p) if cfg["GaussianBlur"] else None,
            A.GlassBlur(p=p) if cfg["GlassBlur"] else None,
            A.HueSaturationValue(p=p) if cfg["HueSaturationValue"] else None,
            A.ISONoise(p=p) if cfg["ISONoise"] else None,
            A.ImageCompression(p=p) if cfg["ImageCompression"] else None,
            A.InvertImg(p=p) if cfg["InvertImg"] else None,
            A.MedianBlur(p=p) if cfg["MedianBlur"] else None,
            A.MotionBlur(p=p) if cfg["MotionBlur"] else None,
            A.MultiplicativeNoise(p=p) if cfg["MultiplicativeNoise"] else None,
            A.Posterize(p=p) if cfg["Posterize"] else None,
            A.RandomBrightnessContrast(p=p) if cfg["RandomBrightnessContrast"] else None,
            A.RGBShift(p=p) if cfg["RGBShift"] else None,
            A.RandomGamma(p=p) if cfg["RandomGamma"] else None,
            A.RandomRain(p=p) if cfg["RandomRain"] else None,
            A.RandomShadow(p=p) if cfg["RandomShadow"] else None,
            A.RandomSnow(p=p) if cfg["RandomSnow"] else None,
            A.RandomSunFlare(p=p) if cfg["RandomSunFlare"] else None,
            A.RandomToneCurve(p=p) if cfg["RandomToneCurve"] else None,
            A.RandomFog(p=p) if cfg["RandomFog"] else None,
            A.RingingOvershoot(p=p) if cfg["RingingOvershoot"] else None,
            A.Sharpen(p=p) if cfg["Sharpen"] else None,
            A.Solarize(p=p) if cfg["Solarize"] else None,
            #add
            #A.ElasticTransform(p=p) if cfg["ElasticTransform"] else None,
            #A.GridDistortion(p=p) if cfg["GridDistortion"] else None,
            #A.OpticalDistortion(p=p) if cfg["OpticalDistortion"] else None,
        ]

        

        self.geomertic_aug = [
            A.HorizontalFlip(p=p) if cfg["HorizontalFlip"] else None,
            A.VerticalFlip(p=p) if cfg["VerticalFlip"] else None,
            A.RandomRotate90(p=p) if cfg["RandomRotate90"] else None,
        ]

        self.distortion_aug = [
            A.ElasticTransform(p=p) if cfg["ElasticTransform"] else None,
            A.GridDistortion(p=p) if cfg["GridDistortion"] else None,
            A.OpticalDistortion(p=p) if cfg["OpticalDistortion"] else None,
        ]



        #filtering out None values
        self.color_augmentations = [aug for aug in self.color_augmentations if aug is not None]
        self.geomertic_aug = [aug for aug in self.geomertic_aug if aug is not None]
        self.distortion_aug = [aug for aug in self.distortion_aug if aug is not None]
        #to albumentations
        self.color_augmentations = A.SomeOf(self.color_augmentations, cfg["number_of_color_aug"])
        self.geomertic_aug = A.Compose(self.geomertic_aug)
        self.distortion_aug = A.SomeOf(self.distortion_aug, cfg["number_of_distortion_aug"])

        #get the right augmentations
        self.number_of_color_aug = cfg["number_of_color_aug"]
        self.number_of_distortion_aug = cfg["number_of_distortion_aug"]


        self.normalize = A.Compose(
            [
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )
        self.resize = A.Compose(
            [
                A.Resize(global_crops_size, global_crops_size),
            ]
        )

        
        logger.info("###################################")
        logger.info("COLOR AUGMENTATIONS")
        logger.info(self.color_augmentations)
        logger.info("GEOMETRIC AUGMENTATIONS")
        logger.info(self.geomertic_aug)
        logger.info("DISTORTION AUGMENTATIONS")
        logger.info(self.distortion_aug)
        logger.info("###################################")

    def __call__(self, image):
        output = {}
        image = cv2.imread(image)
        image = self.resize(image=image)["image"]
        """
        first global crop gets the color augmentations
        second global crop gets the distortion augmentations
        if there is a 3rd global crop, it gets both color and distortion augmentations

        all global crops get the geometric augmentations

        local crops get the geometric augmentations and color augmentations
        
        """
        # global crops:
        im1_base = self.cropping_global(image=image)["image"]
        global_1_aug = self.color_augmentations(image=im1_base)["image"]
        if self.number_of_distortion_aug > 0:
            global_1_aug = self.distortion_aug(image=global_1_aug)["image"]
        global_1_aug = self.geomertic_aug(image=global_1_aug)["image"]
        global_crop_1 = self.normalize(image=global_1_aug)["image"]
        



        im2_base = self.cropping_global(image=image)["image"]
        global_2_aug = self.color_augmentations(image=im2_base)["image"]
        if self.number_of_distortion_aug > 0:
            global_2_aug = self.distortion_aug(image=global_2_aug)["image"]
        global_2_aug = self.geomertic_aug(image=global_2_aug)["image"]
        global_crop_2 = self.normalize(image=global_2_aug)["image"]


        if self.use_3rd_global_crop:
            im3_base = self.cropping_global(image=image)["image"]
            global_3_aug = self.color_augmentations(image=im3_base)["image"]
            global_3_aug = self.distortion_aug(image=global_3_aug)["image"]
            global_3_aug = self.geomertic_aug(image=global_3_aug)["image"]
            global_crop_3 = self.normalize(image=global_3_aug)["image"]
            output["global_crops"] = [global_crop_1, global_crop_2, global_crop_3]
            output["global_crops_teacher"] = [global_crop_1, global_crop_2, global_crop_3]

        else:

            output["global_crops"] = [global_crop_1, global_crop_2]
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]


        # local crops:
        local_crops_intermediate = [
            self.cropping_local(image=image)["image"] for _ in range(self.local_crops_number)
        ]
        local_crops = []
        for local_crop in local_crops_intermediate:
            local_crop = self.color_augmentations(image=local_crop)["image"]
            #local_crop = self.distortion_aug(image=local_crop)["image"]
            local_crop = self.geomertic_aug(image=local_crop)["image"]
            local_crop = self.normalize(image=local_crop)["image"]
            
            local_crops.append(local_crop)
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
    
