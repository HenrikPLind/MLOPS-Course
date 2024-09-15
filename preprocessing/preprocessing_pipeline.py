import numpy as np
import tifffile as tiff
from itertools import product
from PIL import Image
from sklearn.model_selection import train_test_split


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


def preprocessing(input_images, label_images, folder_training, folder_validation, folder_testing):
    # Split the data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(input_images, label_images, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,  test_size=0.2, random_state=42)

    training_patches = patchMaking(X_train, d=256, e=256, overlap=128)
    validation_patches = patchMaking(X_val, d=256, e=256, overlap=128)
    test_patches = patchMaking(X_test, d=256, e=256, overlap=128)


    savePatches(patches=training_patches, path=folder_training, images=images, startImgNo=0)
    savePatches(patches=validation_patches, path=folder_validation, images=images, startImgNo=0)
    savePatches(patches=test_patches, path=folder_testing, images=images, startImgNo=0)


    return training_patches, validation_patches


