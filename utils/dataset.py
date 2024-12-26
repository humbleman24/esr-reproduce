import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None, augment=False):
        super(SRDataset, self).__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        if self.augment:
            if random.random() > 0.5:
                lr_image = lr_image.transpose(Image.FLIP_LEFT_RIGHT)
                hr_image = hr_image.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                lr_image = lr_image.transpose(Image.FLIP_TOP_BOTTOM)
                hr_image = hr_image.transpose(Image.FLIP_TOP_BOTTOM)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image
