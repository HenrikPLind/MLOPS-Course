import os
import pandas as pd
import cv2
import tifffile as tiff


def load_patches(image_dir):
    patches = []
    image_paths = []

    # Create a list of image paths (including .tif files)
    for fname in os.listdir(image_dir):
        if fname.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_path = os.path.join(image_dir, fname)
            image_paths.append(image_path)

            if image_path.endswith((".tif")):

                try:
                    img = tiff.imread(image_path)
                    patches.append(img)

                except Exception as e:
                    print(f"Error opening {image_path}: {e}")

            elif image_path.endswith(".png"):
                try:
                    img = cv2.imread(image_path, 0)
                    patches.append(img)

                except Exception as e:
                    print(f"Error opening {image_path}: {e}")
                    # Save DataFrame to CSV

    return patches