import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CUBDataset(Dataset):
  def __init__(self, root_dir, train=True, transform=None):
    self.root_dir = root_dir
    self.transform = transform

    base_path = os.path.join(root_dir, 'CUB_200_2011')
    image_file = os.path.join(base_path, 'images.txt')
    label_file = os.path.join(base_path, 'image_class_labels.txt')
    split_file = os.path.join(base_path, 'train_test_split.txt')

    # Read files
    images = pd.read_csv(os.path.join(base_path, 'images.txt'), sep=' ', names=['img_id', 'img_path'])
    labels = pd.read_csv(os.path.join(base_path, 'image_class_labels.txt'), sep=' ', names=['img_id', 'label'])
    split = pd.read_csv(os.path.join(base_path, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_train'])

    # Merge info
    data = images.merge(labels, on='img_id')
    data = data.merge(split, on='img_id')

    # Filter train/test
    if train:
      self.data = data[data['is_train'] == 1]
    else:
      self.data = data[data['is_train'] == 0]

    self.base_image_path = os.path.join(base_path, 'images')

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    image_path = os.path.join(self.base_image_path, row['img_path'])
    label = row['label'] - 1
    image = Image.open(image_path).convert("RGB")

    if self.transform:
      image = self.transform(image)

    return image, label

