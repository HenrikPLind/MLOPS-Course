import base64
import io
import json
import subprocess
import requests
from PIL import Image


def serve_mlflow_model(model_uri, port=5001):
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
    Sends an image to the MLflow model server for prediction.

    :param image_path: The path to the image file.
    :param model_url: The URL where the MLflow model is being served.
    :return: The prediction result from the model server.
    """
    try:
        # Construct the model server URL dynamically
        model_url = f"http://{host}:{port}/invocations"
        # Load and preprocess the image
        image = Image.open(image_path)
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

        # Return the prediction result
        return response.json()

    except Exception as e:
        print(f"Error occurred: {e}")
        return None
