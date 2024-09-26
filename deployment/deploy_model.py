import json
import subprocess
import time

import numpy as np
import requests
from PIL import Image
from matplotlib import pyplot as plt


def serve_model(model_uri, port=5001, max_retries=5, wait_time=2):
    """
    Serve an MLflow model using mlflow models serve command and check if it's serving successfully.

    :param model_uri: The URI of the MLflow model to serve (e.g., 'models:/<model_name>/<version_or_stage>')
    :param port: The port number to serve the model on. Default is 5001.
    :param max_retries: Maximum number of retries to check if the server is up.
    :param wait_time: Time to wait between retries (in seconds).
    :return: Process if the model is served successfully, None otherwise.
    """
    try:
        # Construct the command to serve the model
        command = [
            "mlflow", "models", "serve",
            "-m", model_uri,
            "--port", str(port),
            "--no-conda"
        ]

        # Start the MLflow model server
        process = subprocess.Popen(command)
        print(f"Serving model {model_uri} on port {port}...")

        # Check if the server is running by making a request to the /ping endpoint
        url = f"http://localhost:{port}/ping"

        for attempt in range(max_retries):
            try:
                # Wait before making the request
                time.sleep(wait_time)

                # Send a request to check if the model server is up
                response = requests.get(url)

                # If the server responds with status code 200, it's running
                if response.status_code == 200:
                    print(f"Model is successfully serving at {url}")
                    return process
            except requests.ConnectionError:
                print(f"Attempt {attempt + 1}/{max_retries}: Model not yet serving. Retrying...")

        # If max retries are reached and no successful response
        print(f"Failed to serve model {model_uri} on port {port}.")
        process.terminate()  # Terminate the process if it is not running correctly
        return None

    except Exception as e:
        print(f"Failed to serve the model: {e}")
        return None


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
        """
        JSON Serialization: NumPy arrays are not directly JSON-serializable because they are not native Python data types. 
        JSON serialization requires data to be in standard Python types like lists, dictionaries, integers, floats, and strings.
        Compatibility with MLflow Model Server: The MLflow model server expects the input data to be in JSON format, 
        typically under the "inputs" key. Converting the NumPy array to a list ensures that the image data can be serialized 
        into JSON and sent in the correct format.
        """
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