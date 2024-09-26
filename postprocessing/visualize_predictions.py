import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_segmentation_map(class_colors, segmentation_map, save_path=None):
    """
    Visualizes the segmentation map and saves it to a file if save_path is provided.

    :param segmentation_map: NumPy array of shape (height, width) with class labels.
    :param save_path: Optional; Path to save the segmentation image.
    """

    # Create an RGB image to display
    height, width = segmentation_map.shape
    segmentation_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        # Create a mask for the current class
        mask = segmentation_map == class_id
        segmentation_image[mask] = color

    # Convert the segmentation image to a PIL image
    segmentation_pil = Image.fromarray(segmentation_image)

    # Save the image if a path is provided
    if save_path:
        segmentation_pil.save(save_path)
        print(f"Segmentation map saved to {save_path}")

    # Display the segmentation image
    plt.figure(figsize=(10, 10))
    plt.imshow(segmentation_image)
    plt.axis('off')
    plt.title('Segmentation Map')
    plt.show()

def overlay_segmentation(original_image_path, segmentation_map, class_colors, alpha=0.5, save_path=None):
    """
    Overlays the segmentation map on the original image and saves it if save_path is provided.

    :param original_image_path: Path to the original image file.
    :param segmentation_map: 2D NumPy array with class labels.
    :param class_colors: Dictionary mapping class IDs to RGB colors.
    :param alpha: Transparency factor for the overlay.
    :param save_path: Optional; Path to save the overlayed image.
    """
    # Load the original image
    original_image = Image.open(original_image_path)
    original_image = np.array(original_image)

    # Ensure the original image and segmentation map have the same dimensions
    if original_image.shape[:2] != segmentation_map.shape:
        # Resize the segmentation map to match the original image
        segmentation_map_resized = Image.fromarray(segmentation_map.astype(np.uint8)).resize(
            (original_image.shape[1], original_image.shape[0]),
            resample=Image.NEAREST
        )
        segmentation_map = np.array(segmentation_map_resized)

    # Create the segmentation image
    segmentation_image = np.zeros_like(original_image)
    for class_id, color in class_colors.items():
        mask = segmentation_map == class_id
        segmentation_image[mask] = color

    # Blend the images
    blended_image = ((1 - alpha) * original_image + alpha * segmentation_image).astype(np.uint8)

    # Convert blended image to PIL image
    blended_pil = Image.fromarray(blended_image)

    # Save the image if a path is provided
    if save_path:
        blended_pil.save(save_path)
        print(f"Overlayed image saved to {save_path}")

    # Display the overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(blended_image)
    plt.axis('off')
    plt.title('Overlayed Segmentation')
    plt.show()

def load_ground_truth_mask(ground_truth_path):
    """
    Loads the ground truth segmentation mask.

    :param ground_truth_path: Path to the ground truth segmentation mask image.
    :return: A 2D NumPy array containing class labels for each pixel.
    """
    try:
        # Load the ground truth image
        ground_truth_image = Image.open(ground_truth_path).convert('L')  # Convert to grayscale
        ground_truth_array = np.array(ground_truth_image)

        # If your ground truth uses color codes instead of class labels,
        # you'll need to map colors to class labels.

        return ground_truth_array

    except Exception as e:
        print(f"Error loading ground truth mask: {e}")
        return None

def compare_segmentation_results(original_image_path, ground_truth_path, segmentation_map, class_colors, save_path=None):
    """
    Displays and saves the comparison between ground truth and predicted segmentation.

    :param original_image_path: Path to the original image.
    :param ground_truth_path: Path to the ground truth segmentation mask.
    :param segmentation_map: Predicted segmentation map as a 2D NumPy array.
    :param class_colors: Dictionary mapping class IDs to RGB colors.
    :param save_path: Optional; Path to save the comparison image.
    """
    try:
        # Load the original image
        original_image = Image.open(original_image_path).convert('RGB')
        original_image_array = np.array(original_image)

        # Load the ground truth segmentation mask
        ground_truth_array = load_ground_truth_mask(ground_truth_path)
        if ground_truth_array is None:
            print("Ground truth segmentation mask could not be loaded.")
            return

        # Ensure all images are the same size
        height, width = original_image_array.shape[:2]

        # Resize ground truth if necessary
        if ground_truth_array.shape != (height, width):
            ground_truth_array = np.array(
                Image.fromarray(ground_truth_array).resize((width, height), resample=Image.NEAREST)
            )

        # Create colored ground truth segmentation image
        ground_truth_image = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            mask = ground_truth_array == class_id
            ground_truth_image[mask] = color

        # Create colored predicted segmentation image
        predicted_image = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            mask = segmentation_map == class_id
            predicted_image[mask] = color

        # Create overlayed images
        alpha = 0.5  # Transparency factor

        ground_truth_overlay = ((1 - alpha) * original_image_array + alpha * ground_truth_image).astype(np.uint8)
        predicted_overlay = ((1 - alpha) * original_image_array + alpha * predicted_image).astype(np.uint8)

        # Display the images side by side
        fig, axes = plt.subplots(1, 4, figsize=(20, 10))

        axes[0].imshow(original_image_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(ground_truth_image)
        axes[1].set_title('Ground Truth Segmentation')
        axes[1].axis('off')

        axes[2].imshow(predicted_image)
        axes[2].set_title('Predicted Segmentation')
        axes[2].axis('off')

        axes[3].imshow(predicted_overlay)
        axes[3].set_title('Overlayed Prediction')
        axes[3].axis('off')

        plt.tight_layout()

        # Save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Comparison image saved to {save_path}")

        plt.show()

    except Exception as e:
        print(f"Error comparing segmentation results: {e}")
