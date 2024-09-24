import base64
import io
import json
import subprocess

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
        # Construct the command to serve the model: mlflow models serve -m models:/my_model/1 --port 5000
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
    Sends an image to the MLflow model server for prediction and visualizes the segmentation result.

    :param image_path: The path to the image file.
    :param port: The port on which the MLflow model is being served.
    :param host: The host where the MLflow model is being served. Default is localhost (127.0.0.1).
    :return: The prediction result from the model server.
    """
    try:
        # Construct the model server URL dynamically
        model_url = f"http://{host}:{port}/invocations"

        # Load and preprocess the image
        image = Image.open(image_path)
        original_image = image.copy()  # Save a copy of the original for visualization
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Encode the image as base64
        encoded_image = base64.b64encode(img_bytes).decode('utf-8')

        # Prepare the JSON payload
        data = json.dumps({"image": encoded_image})

        # Send the request to the MLflow model server
        response = requests.post(
            url=model_url,
            headers={"Content-Type": "application/json"},
            data=data
        )

        # Check if the request was successful
        response.raise_for_status()

        # Parse the response (assuming the segmentation is returned as a base64-encoded image)
        response_json = response.json()
        segmentation_mask_base64 = response_json.get('figure_out_the_key_to_image')

        if segmentation_mask_base64:
            # Decode the segmentation mask from base64
            segmentation_mask_bytes = base64.b64decode(segmentation_mask_base64)
            segmentation_image = Image.open(io.BytesIO(segmentation_mask_bytes))

            # Visualize the segmentation result
            visualize_segmentation(original_image, segmentation_image)
        else:
            print("No segmentation mask found in the response.")

        return response_json

    except Exception as e:
        print(f"Error occurred: {e}")
        return None



def visualize_segmentation(original_image, segmentation_image):
    """
    Visualizes the original image with the segmentation mask overlay.

    :param original_image: The original input image.
    :param segmentation_image: The predicted segmentation mask.
    """
    # Convert images to numpy arrays
    original_array = np.array(original_image)
    segmentation_array = np.array(segmentation_image)

    # Ensure segmentation mask is binary or in a compatible format
    if segmentation_array.ndim == 2:  # Single channel mask
        plt.imshow(original_array)
        plt.imshow(segmentation_array, alpha=0.5, cmap='jet')  # Overlay segmentation with transparency
        plt.title("Segmentation Overlay")
    else:
        # If segmentation image is multi-channel (RGB), we can directly overlay it
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(segmentation_image)
        plt.title("Segmentation Result")

    plt.show()