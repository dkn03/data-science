import os
import torch

from torchvision.io import read_image
from torch.utils.data import Dataset

class CardDataset(Dataset):
    def __init__(self, image_names: str, y, path, transforms=None, normalize=True):
        self.x, self.y = image_names, y
        self.path = path
        self.normalize = normalize
        self.transforms = transforms
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        image_name = self.x[index]
        image = read_image(self.path + image_name).to(dtype=torch.float)
        label = self.y[index]
        if self.transforms:
            image, label = self.transforms(image, label)
        if self.normalize:
            image = image / 255
        
        return image, label

def extract_data(path: str, mode: str = 'all'):
    folders = sorted(os.listdir(path))
    images = []
    labels = []
    suits = ['clubs', 'diamonds', 'hearts', 'spades', 'joker']
    values = ['eight', 'five', 'four', 'jack', 'joker', 'king', 'nine', 'queen', 'seven', 'six', 'ten', 'three', 'two', 'ace']
    for i, folder in enumerate(folders):
        match mode:
            case 'all':
                label = torch.zeros(len(folders))
                label[i] = 1
            case 'suits':
                label = torch.tensor(list(map(lambda x: folder.endswith(x), suits))).to(dtype=torch.float)
            case 'values':
                label = torch.tensor(list(map(lambda x: folder.startswith(x), values))).to(dtype=torch.float)
        
        for image in os.listdir(f"{path}/{folder}"):
            images.append(f"{folder}/{image}")        
            labels.append(label)
            #print(images[-1])

    return images, labels