# File for loading images and patch making!

#### Function for loading images from folder ####
from PIL import Image
import os

def loadImages(path):
    images = []
    for filename in os.listdir(path):
        try:
            image = Image.open(os.path.join(path, filename))
            if image is not None:
                images.append(image)
                print('Loaded image ' + filename)
        except:
            print('FAIL: Can not load ' + filename)
    return images

# Paths for folders
pathTrainOrig = 'C:/Users/matia/Desktop/ColorPatchData/TrainingInputData' # Matias
pathTrainOutput = 'C:/Users/matia/Desktop/ColorPatchData/TrainingOutputData' # Matias
pathValidOrig = 'C:/Users/matia/Desktop/ColorPatchData/ValidationInputData'# Matias
pathValidOutput = 'C:/Users/matia/Desktop/ColorPatchData/ValidationOutputData' # Matias

#pathTrainOrig = 'C:/Users/annem/OneDrive - Aalborg Universitet/9. semester/ST9Code/CodeST9/Data_for_network/TrainingData/TrainingInputData' # Anne
#pathTrainOutput = 'C:/Users/annem/OneDrive - Aalborg Universitet/9. semester/ST9Code/CodeST9/Data_for_network/TrainingData/TrainingOutputData' # Anne
#pathValidOrig = 'C:/Users/annem/OneDrive - Aalborg Universitet/9. semester/ST9Code/CodeST9/Data_for_network/ValidationData/ValidationInputData' # Anne
#pathValidOutput = 'C:/Users/annem/OneDrive - Aalborg Universitet/9. semester/ST9Code/CodeST9/Data_for_network/ValidationData/ValidationOutputData' # Anne

# Execution of function for loading images
print('#### Loading images... ####')
imagesTrainOrig = loadImages(path=pathTrainOrig)
print('Number of loaded original images for training: ', len(imagesTrainOrig))

imagesTrainOutput = loadImages(path=pathTrainOutput)
print('Number of loaded output images for training: ', len(imagesTrainOutput))

imagesValidOrig = loadImages(path=pathValidOrig)
print('Number of loaded original images for validation: ', len(imagesValidOrig))

imagesValidOutput = loadImages(path=pathValidOutput)
print('Number of loaded output images for validation: ', len(imagesValidOutput))

#### Function for patch making ####
from itertools import product

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

# Execution of function for patch making
print('#### Patch making... ####')
patchesTrainOrig = patchMaking(images=imagesTrainOrig, d=256, e=256, overlap=128)
print('Number of patches of original images for training: ', len(patchesTrainOrig))

patchesTrainOutput = patchMaking(images=imagesTrainOutput, d=256, e=256, overlap=128)
print('Number of patches of output images for training: ', len(patchesTrainOutput))

patchesValidOrig = patchMaking(images=imagesValidOrig, d=256, e=256, overlap=128)
print('Number of patches of original images for validation: ', len(patchesValidOrig))

patchesValidOutput = patchMaking(images=imagesValidOutput, d=256, e=256, overlap=128)
print('Number of patches of output images for validation: ', len(patchesValidOutput))

#### Save patches to folder ####
from PIL import Image

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

# Paths for folders
pathTrainOrigPatches = 'C:/Users/matia/Desktop/ColorPatchData/TrainDataInput_Patches/'
pathTrainOutputPatches = 'C:/Users/matia/Desktop/ColorPatchData/TrainDataMask_Patches/'
pathValidOrigPatches = 'C:/Users/matia/Desktop/ColorPatchData/TrainDataInput_Patches/'
pathValidOutputPatches = 'C:/Users/matia/Desktop/NoColorPatchData/TrainDataMask_Patches/'

#pathTrainOrigPatches = 'C:/Users/annem/OneDrive - Aalborg Universitet/9. semester/PatchesOverlap/TrainingData/TrainingInputData/'
#pathTrainOutputPatches = 'C:/Users/annem/OneDrive - Aalborg Universitet/9. semester/PatchesOverlap/TrainingData/TrainingOutputData/'
#pathValidOrigPatches = 'C:/Users/annem/OneDrive - Aalborg Universitet/9. semester/PatchesOverlap/ValidationData/ValidationInputData/'
#pathValidOutputPatches = 'C:/Users/annem/OneDrive - Aalborg Universitet/9. semester/PatchesOverlap/ValidationData/ValidationOutputData/'

print('#### Saving patches to folders... ####')
savePatches(patches=patchesTrainOrig, path=pathTrainOrigPatches, images=imagesTrainOrig, startImgNo=0)
savePatches(patches=patchesTrainOutput, path=pathTrainOutputPatches, images=imagesTrainOutput, startImgNo=0)
savePatches(patches=patchesValidOrig, path=pathValidOrigPatches, images=imagesValidOrig, startImgNo=18)
savePatches(patches=patchesValidOutput, path=pathValidOutputPatches, images=imagesValidOutput, startImgNo=18)

print('DONE')