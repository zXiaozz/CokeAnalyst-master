# -*- coding: utf-8 -*-
# @Time    : 20/12/25 9:22
# @Author  : Lazycatt
# @File    : coke_dataloader.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch
import cv2
# import albumentations as A


class MyDataset(Dataset):
    def __init__(self, imageDir, labelDir, edgeDir, transform, fold, nFolds=5, train=True):
        self.imageDir = imageDir
        self.labelDir = labelDir
        self.edgeDir = edgeDir
        self.transform = transform

        names = np.array(os.listdir(imageDir))
        kf = KFold(n_splits=nFolds, shuffle=True)
        names = names[list(kf.split(names))[fold][0 if train else 1]]
        self.ids = names

    def __getitem__(self, index):
        name = self.ids[index]

        image = self.rgb_loader(os.path.join(self.imageDir, name))
        label = self.binary_loader(os.path.join(self.labelDir, name))
        edge = self.binary_loader(os.path.join(self.edgeDir, name))

        image = self.transform(image)
        # image = transforms.Normalize(mean=[0.272, 0.201, 0.502], std=[0.143, 0.124, 0.198])(image)

        label = self.transform(label)
        edge = self.transform(edge)

        return image, label, edge

    def __len__(self):
        return len(self.ids)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')


def TrainTransform(size=768):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


def ValidTransform(size=768):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class TestDataset:
    def __init__(self, imageDir, imageSize):
        self.imageDir = imageDir
        self.imageSize = imageSize
        self.images = os.listdir(imageDir)
        self.transform = transforms.Compose([
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(os.path.join(self.imageDir, self.images[self.index]))
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index]

        self.index += 1

        return image, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

