import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader


device = ("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv('train.csv', names=["img_name", "label"])
val_df = pd.read_csv('val.csv', names=["img_name", "label"])
test_df = pd.read_csv('test.csv', names=["img_name", "label"])

# Normalize outputs to [-1, 1]
def normalize(x, xmin=0, xmax=36):
    return 2 * ((x - xmin) / (xmax - xmin)) - 1

class OCTslices(Dataset):
    def __init__(self, annotation_file, base_dir, transform=None):
        self.base_dir = base_dir
        self.annotations = annotation_file
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img_dir = os.path.join(self.base_dir, str(img_id))
        img = Image.open(img_dir)

        y_label = float(self.annotations.iloc[index, 1])

        # Apply the standard transformation first (convert to tensor and normalize)
        if self.transform is not None:
            img = self.transform(img)

        # Apply light augmentation for high-frequency data (sensitivity >= 20)
        if y_label >= 20:
            img = self._apply_light_augmentation(img)

        # Apply strong augmentation for low-sensitivity data (sensitivity < 20) with 50% probability
        if y_label < 20 and torch.rand(1).item() < 0.5:
            img = self._apply_strong_augmentation(img)

        y_label = torch.tensor(y_label)
        return img, y_label

    def _apply_light_augmentation(self, img):
        """Applies light augmentation for high-frequency images (sensitivity >= 20)"""
        transform_light = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation((-5, 5))  # Light rotation
                ],
                p=0.75  # 75% probability of applying light augmentation
            )
        ])
        return transform_light(img)

    def _apply_strong_augmentation(self, img):
        """Applies heavy augmentation for low-sensitivity images (sensitivity < 20)"""
        transform_strong = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation((-8, 8)),
                    # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=(-1, 1))
                ],
                p=0.75  # 75% probability of applying heavy augmentation
            )
        ])
        return transform_strong(img)

def getTrain(norm=False, num_chan=1, batch_size=32):
    assert num_chan == 1
    if norm:
        train_df['label'] = normalize(train_df['label'].values)
    
    # No need for strong_augment anymore, augmentation is handled in __getitem__
    trainset = OCTslices(train_df, base_dir='/home/phiat/Dissertation/new_yl',
                         transform=train_transform) 
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader

def getVal(norm=False, num_chan=1):
    assert num_chan == 1
    if norm:
        val_df['label'] = normalize(val_df['label'].values)
    valset = OCTslices(val_df, base_dir='/home/phiat/Dissertation/new_yl', transform=eval_transform)
    val_loader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=2)
    return val_loader

def getTest(num_chan=1):
    assert num_chan == 1
    testset = OCTslices(test_df, base_dir='/home/phiat/Dissertation/new_yl', transform=eval_transform)
    test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)
    return test_loader

# Edited Transforms for Train, Val, Test
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize((496, 32), antialias=None),
    transforms.Normalize(mean=[0.1209], std=[0.1118])
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize((496, 32), antialias=None),
    transforms.Normalize(mean=[0.1209], std=[0.1118])
])

# Data Loaders
train_loader = getTrain(norm=False, num_chan=1)  ## True
val_loader = getVal(norm=True, num_chan=1)
test_loader = getTest(num_chan=1)
