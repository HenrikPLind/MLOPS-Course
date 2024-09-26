import base64
import io
import json
import subprocess
import mlflow.keras
import numpy as np
import requests
from PIL import Image
from matplotlib import pyplot as plt


def serve_model(model_uri, port=5001):
    """
    Serve an MLflow model using mlflow models serve command.

    :param model_uri: The URI of the MLflow model to serve (e.g., 'models:/<model_name>/<version_or_stage>')
    :param port: The port number to serve the model on. Default is 5001.
    """
    try:
        # Construct the command to serve the model: mlflow models serve -m models:/my_model/1 --port 5001
        command = [
            "mlflow", "models", "serve",
            "-m", model_uri,
            "--port", str(port)
        ]

        # Start the MLflow model server
        process = subprocess.Popen(command)

        print(f"Serving model {model_uri} on port {port}...")

        # Return the process in case you want to interact with it later (e.g., terminate)
        return process

    except Exception as e:
        print(f"Failed to serve the model: {e}")


def predict_on_deployed_model(image_path, port=5001, host="127.0.0.1"):
    """
    Sends an image to the MLflow model server for prediction.

    :param image_path: The path to the image file.
    :param port: The port on which the MLflow model is being served.
    :param host: The host where the MLflow model is being served. Default is localhost (127.0.0.1).
    :return: The JSON response from the model server, assumed to contain the segmentation result.
    """
    try:
        # Construct the model server URL dynamically
        model_url = f"http://{host}:{port}/invocations"

        image = Image.open(image_path)
        image_array = np.array(image).tolist()  # Convert to list for JSON serialization
        data = json.dumps({"inputs": [image_array]})


        # Send the request to the MLflow model server
        response = requests.post(
            url=model_url,
            headers={"Content-Type": "application/json"},
            data=data
        )

        # Check if the request was successful
        response.raise_for_status()

        # Return the full response as JSON
        return response.json()

    except Exception as e:
        print(f"Error occurred during prediction: {e}")
        return None


def visualize_volume(segmentation_volume):
    """
    Visualizes a 3D volume or 2D segmentation returned from the model.

    :param segmentation_volume: The volume data (as a numpy array or similar) to visualize.
    """
    if isinstance(segmentation_volume, list):
        segmentation_volume = np.array(segmentation_volume)  # Convert to numpy array if it's a list

    # Check if it's 2D or 3D data
    if segmentation_volume.ndim == 2:
        # 2D Image
        plt.imshow(segmentation_volume, cmap='jet', alpha=0.5)
        plt.title("Segmentation Mask")
        plt.show()
    elif segmentation_volume.ndim == 3:
        # 3D Volume: Visualize each slice
        fig, axes = plt.subplots(1, segmentation_volume.shape[0], figsize=(15, 5))
        fig.suptitle("Segmentation Volume Slices")
        for i, ax in enumerate(axes):
            ax.imshow(segmentation_volume[i], cmap='jet', alpha=0.5)
            ax.set_title(f'Slice {i + 1}')
        plt.show()
    else:
        print("Unsupported volume shape for visualization.")