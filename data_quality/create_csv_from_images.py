import os
import pandas as pd
import cv2
import tifffile as tiff

def load_images_from_folder(image_dir):
    image_paths = []
    input_images = []
    # Create a list of image paths (including .tif files)
    for fname in os.listdir(image_dir):
        if fname.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_path = os.path.join(image_dir, fname)
            image_paths.append(image_path)
            if image_path.endswith((".png")):

                try:
                    img = cv2.imread(image_path)
                    input_images.append(img)
                except Exception as e:
                    print(f"Error opening {image_path}: {e}")
    return input_images


def process_images_and_save_as_csv(image_dir, csv_filename):
    input_images = []
    image_x_arr = []
    image_y_arr = []
    image_z_arr = []
    variance_arr = []
    image_paths = []

    # Create a list of image paths (including .tif files)
    for fname in os.listdir(image_dir):
        if fname.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_path = os.path.join(image_dir, fname)
            image_paths.append(image_path)

            if image_path.endswith((".tif")):

                try:
                    img = tiff.imread(image_path)
                    image_x, image_y, image_z = img.shape

                    # Apply the Laplacian filter
                    laplacian = cv2.Laplacian(img, cv2.CV_64F)

                    # Calculate the variance of the Laplacian
                    variance = laplacian.var()
                    variance_arr.append(variance)
                    image_x_arr.append(image_x)
                    image_y_arr.append(image_y)
                    image_z_arr.append(image_z)
                    input_images.append(img)

                    # Create a DataFrame
                    df = pd.DataFrame({
                        'image_x_res': image_x_arr,
                        'image_y_res': image_y_arr,
                        'image_z_res': image_z_arr,
                        'image_path': image_paths,
                        'blurriness': variance_arr
                    })
                except Exception as e:
                    print(f"Error opening {image_path}: {e}")

            elif image_path.endswith(".png"):
                try:
                    img = cv2.imread(image_path, 0)
                    image_x, image_y = img.shape

                    # Apply the Laplacian filter
                    laplacian = cv2.Laplacian(img, cv2.CV_64F)

                    # Calculate the variance of the Laplacian
                    variance = laplacian.var()
                    variance_arr.append(variance)
                    image_x_arr.append(image_x)
                    image_y_arr.append(image_y)
                    input_images.append(img)

                    df = pd.DataFrame({
                        'image_x_res': image_x_arr,
                        'image_y_res': image_y_arr,
                        'image_path': image_paths,
                        'blurriness': variance_arr
                    })
                except Exception as e:
                    print(f"Error opening {image_path}: {e}")
                    # Save DataFrame to CSV
    df.to_csv(csv_filename, index=False)
    print(f"CSV file saved as {csv_filename}")
    return input_images



