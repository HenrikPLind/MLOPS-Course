import numpy as np
import tifffile as tiff
from patchify import patchify

def preprocessing(images):
    for i in range(len(images)):
        current_image = images[i]
        # Normalize images
        current_image = current_image/255

        # patchify image

        patches = patchify(current_image, (256,256), step=256)

        return patches


