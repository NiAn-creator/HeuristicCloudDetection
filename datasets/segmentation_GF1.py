import os
import torch.utils.data as data
import numpy as np
import torch


class wscd_train(data.Dataset):

    def __init__(self, root, mask_dir, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]

        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]

        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):

        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)

        weakLable = np.load(self.masks[index])
        weakLable = np.asarray(weakLable, dtype=np.float32)
        weakLable = np.expand_dims(weakLable, axis=0)

        hedLable = np.zeros((321,321),dtype=np.float32)
        hedLable[weakLable>0]=1

        img = torch.Tensor(rsData[:,:-1,:-1])
        weakLable = torch.tensor(weakLable[:,:-1,:-1])
        hedLable = torch.tensor(hedLable[:,:-1,:-1])

        mean = torch.tensor([493.98721723, 478.55947529, 424.43626955, 520.79453991])
        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        return img, weakLable, hedLable

    def __len__(self):
        return len(self.images)

class wscd_trainval(data.Dataset):

    def __init__(self, root, image_set='trainval'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')

        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if not os.path.exists(image_dir):
            raise ValueError(
                'Wrong image_dir entered!')
        if not os.path.exists(mask_dir):
            raise ValueError(
                'Wrong mask_dir entered!')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]

        assert (len(self.images) == len(self.masks))


    def __getitem__(self, index):
        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData,dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)

        mask = np.load(self.masks[index])
        mask = np.asarray(mask, dtype=np.float32)

        img = torch.Tensor(rsData[:,:-1,:-1])
        target = torch.Tensor(mask[:-1,:-1])

        mean = torch.tensor([547.20599439, 514.92978335, 433.80227233, 529.83716347])
        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        return img, target

    def __len__(self):
        return len(self.images)

class wscd_test(data.Dataset):

    def __init__(self, root, image_set='test'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')

        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if not os.path.exists(image_dir):
            raise ValueError(
                'Wrong image_dir entered!')
        if not os.path.exists(mask_dir):
            raise ValueError(
                'Wrong mask_dir entered!')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData,dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)

        mask = np.load(self.masks[index])
        mask = np.asarray(mask, dtype=np.float32)

        img = torch.Tensor(rsData[:,:-1,:-1])
        target = torch.Tensor(mask[:-1,:-1])

        mean = torch.tensor([493.66686932, 500.62488985, 460.50668585, 612.65028453])
        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        return img, target

    def __len__(self):
        return len(self.images)
