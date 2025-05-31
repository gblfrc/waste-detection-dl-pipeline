import json
import numpy as np
import os
import sys
import torch
import torchvision.transforms.v2 as transforms

sys.path.append('..')

from PIL import Image
from torch.utils.data import Dataset
from misc.utils import onehot

class AWClassificationDataset(Dataset):
    '''
    Dataset for binary classification. Assumes dataset information is stored in a similar way as for the AerialWaste dataset. 
    Therefore, the implementation assumes:
    - general information about the dataset is contained in a json file with an "images" field, storing all information about images
    - each image in the list at the previous point has a "file_name" attribute defining the image position with respect to a specific image folder (passed as construction parameter)
    '''
    def __init__(self, json_path, img_folder, transform=None):
        # Try loading json file
        with open(json_path, 'r') as file:
            content = json.load(file)
        # Save dataset information as object attributes
        self.categories = content['categories']
        self.cat_ids = [cat['id'] for cat in self.categories]       # used for 1-hot encoding
        self.cat_names = [cat['name'] for cat in self.categories]   # might be needed for debugging or plotting
        self.images = [(os.path.join(img_folder, img['file_name']), onehot(img['categories'], self.cat_ids)) for img in content['images']]
        self.transform = transform
        self.toTensor = transforms.Compose([transforms.ToImage(), 
                                            transforms.ToDtype(torch.float32, scale=True)])
        # Sanity check on dataset images
        for img in self.images:
            assert(os.path.isfile(img[0])) # check image exists

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Get image data from list of images
        img_path = self.images[index][0]
        label = self.images[index][1]
        # Read image
        img = Image.open(img_path).convert('RGB') # convert to RGB if RGBA or other
        # Convert images and labels to tensors
        img = self.toTensor(img)
        label = torch.tensor(label)
        # Apply transform if required
        if self.transform:
            img = self.transform(img)

        return img, label
    
class AWClassificationDatasetNamed(AWClassificationDataset):
    '''
    Dataset for binary classification. Assumes dataset information is stored in a similar way as for the AerialWaste dataset. 
    Therefore, the implementation assumes:
    - general information about the dataset is contained in a json file with an "images" field, storing all information about images
    - each image in the list at the previous point has a "file_name" attribute defining the image position with respect to a specific image folder (passed as construction parameter)
    The dataset differs from its parent in the values returned when __getitem__ is called. This dataset, beside the image as a tensor and the
    associated label, also provides the name of the extracted image.
    '''
    def __init__(self, json_path, img_folder, transform=None):
        # Try loading json file
        super().__init__(json_path=json_path, img_folder=img_folder, transform=transform)

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        # Get image and label from parent function
        img, label = super().__getitem__(index)
        # Get image data from list of images
        img_path = self.images[index][0]
        name = img_path.split('/')[-1]

        return img, label, name
    

