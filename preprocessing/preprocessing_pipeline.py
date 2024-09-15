import numpy as np
import tifffile as tiff
from patchify import patchify
from itertools import product
from PIL import Image
def patchMaking(images, d, e, overlap):
    patches = []
    for image in images:
        w, h = image.size
        grid = list(product(range(0, h - h % d, overlap), range(0, w - w % e, overlap)))
        for i, j in grid:
            box = (j, i, j + d, i + e)
            patch = image.crop(box)
            patches.append(patch)
    return patches

def savePatches(patches, path, images, startImgNo):
    filename = list()
    for x in range(len(images)):  # Antallet af originale eller output billeder
        for y in range(len(patches) // len(images)):  # Antallet af splits pr billede
            name = "{0}{1}{2}{3}.png".format("image", x + 1 + startImgNo, "split", y + 1)
            # filename = 'image ' + '%06d' + ' split' + ' %06d' % x, y  # image1 split1
            filename.append(name)
    i = 0
    for patch in patches:
        patch.save(path + filename[i])
        i = i + 1

def preprocessing(images):
    patches = patchMaking(images, d=256, e=256, overlap=128)

    savePatches(patches=patches, path='', images=images, startImgNo=0)

    return patches


