import numpy as np
import random
import torchvision.transforms.v2 as transforms

class Rotate90s:
    '''
    A custom transform to apply random rotation of only multiples of 90 degrees.
    Implementation based on: 
        https://github.com/pytorch/vision/issues/566#issuecomment-535854734
    '''
    def __init__(self):
        # Store in an attribute the multiples of 90 [0, 90, 180, 270]
        self.angles = list(90*np.array(range(0,4)))

    def __call__(self, image, *args, **kwargs):
        # Randomly select a multiple of 90 degrees
        angle = random.choice(self.angles)
        # Apply default rotations of the given angle
        return transforms.functional.rotate(image, angle, *args, **kwargs)
