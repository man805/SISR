import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, transform):
        """
        Args:
            root: image directory.
        """
        self.root = root
        self.transform = transform
        self.count = len([name for name in os.listdir(os.path.join(self.root, 'LR')) if os.path.isfile(os.path.join(self.root, 'LR', name))])

    def __getitem__(self, index):
        """Returns one data pair (HR image and LR image)."""
        path = '%04d.jpg' % (index + 1) 
        hr_image = Image.open(os.path.join(self.root, 'HR', path)).convert('RGB')
        lr_image = Image.open(os.path.join(self.root, 'LR', path)).convert('RGB')
        if self.transform is not None:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        return hr_image, lr_image

    def __len__(self):
        return self.count

'''
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths
'''

def get_loader(root, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root, transform=transform)
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader