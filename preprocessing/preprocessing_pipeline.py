import numpy as np
import tifffile as tiff
from itertools import product
from PIL import Image
from sklearn.model_selection import train_test_split

def patchMaking(images, d, e, overlap):
    patches = []
    for image in images:

        w = image.shape[0]
        h = image.shape[1]

        grid = list(product(range(0, h - h % d, overlap), range(0, w - w % e, overlap)))
        for i, j in grid:
            box = (j, i, j + d, i + e)
            # Convert the numpy array to a PIL Image
            patch = image[box[0]:box[2], box[1]:box[3]]
            # Only append patches that match the desired size (d x e)
            if patch.shape[0] == d and patch.shape[1] == e:
                patches.append(patch)
    patches = np.array(patches)
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
        patch_image = Image.fromarray(patch)
        patch_image.save(path + filename[i])
        i = i + 1


def preprocessing(input_images, label_images,
                  folder_training, folder_training_label,
                  folder_validation, folder_validation_label,
                  folder_testing, folder_testing_label):
    # Split the data into training, validation and testing sets
    X_train, X_val, y_train, y_val = train_test_split(input_images, label_images, test_size=0.5, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,  test_size=0.2, random_state=42)

    '### Create training input and label patches ###'
    training_patches = patchMaking(X_train, d=256, e=256, overlap=128)
    training_label_patches = patchMaking(y_train, d=256, e=256, overlap=128)

    '### Create validation input and label patches ###'
    validation_patches = patchMaking(X_val, d=256, e=256, overlap=128)
    validation_label_patches = patchMaking(y_val, d=256, e=256, overlap=128)

    '### Create test input and label patches ###'
    test_patches = patchMaking(X_test, d=256, e=256, overlap=128)
    test_label_patches = patchMaking(y_test, d=256, e=256, overlap=128)

    '### Save training input and label patches ###'
    savePatches(patches=training_patches, path=folder_training, images=X_train, startImgNo=0)
    savePatches(patches=training_label_patches, path=folder_training_label, images=y_train, startImgNo=0)

    '### Save training input and label patches ###'
    savePatches(patches=validation_patches, path=folder_validation, images=X_val, startImgNo=0)
    savePatches(patches=validation_label_patches, path=folder_validation_label, images=y_val, startImgNo=0)

    '### Save training input and label patches ###'
    savePatches(patches=test_patches, path=folder_testing, images=X_test, startImgNo=0)
    savePatches(patches=test_label_patches, path=folder_testing_label, images=y_test, startImgNo=0)


    return training_patches, training_label_patches, \
           validation_patches, validation_label_patches, \
           test_patches, test_label_patches


