import mlflow.keras
import numpy as np


def dice_coefficient_per_class(y_true, y_pred, class_label, smooth=1e-6):
    """
    Calculate Dice coefficient for a specific class.

    Args:
    y_true (numpy array): Ground truth segmentation.
    y_pred (numpy array): Predicted segmentation.
    class_label (int): The label of the class to calculate Dice for.
    smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    float: Dice coefficient for the given class.
    """
    y_true_class = (y_true == class_label).astype(np.float32)
    y_pred_class = (y_pred == class_label).astype(np.float32)

    intersection = np.sum(y_true_class * y_pred_class)
    dice = (2. * intersection + smooth) / (np.sum(y_true_class) + np.sum(y_pred_class) + smooth)

    return dice


def evaluate_segmentation(image, ground_truth, experiment_id, run_id, num_classes=3):
    """
    Evaluate a segmentation model by calculating the predicted segmentation mask and
    comparing it to the ground truth segmentation. Calculates Dice and IoU per class.

    Args:
    model (tf.keras.Model): The pre-trained segmentation model.
    image (numpy array): The input image to the model.
    ground_truth (numpy array): The ground truth segmentation mask.
    num_classes (int): Number of classes (including background).

    Returns:
    dict: A dictionary with Dice and IoU scores per class.
    """
    model_uri = f"../src/mlartifacts/{experiment_id}/{run_id}/artifacts/mlartifacts/model"
    print(f'Fetching model from: {model_uri}')
    model = mlflow.tensorflow.load_model(model_uri)

    # Perform model inference to get predicted mask
    predicted_mask = model.predict(np.expand_dims(image, axis=0))[0]  # Assuming input is single image

    # Convert predictions to class labels
    predicted_mask = np.argmax(predicted_mask, axis=-1)
    predicted_GT = np.argmax(ground_truth, axis=-1)

    # Initialize results dictionary
    results = {"dice": {}, "iou": {}}

    # Loop through each class (including background)
    for class_label in range(num_classes):
        # Calculate Dice coefficient for the class
        dice = dice_coefficient_per_class(predicted_GT, predicted_mask, class_label)

        # Store results
        results["dice"][f"class_{class_label}"] = dice

    return results



