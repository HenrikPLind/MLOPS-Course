import mlflow.keras
import numpy as np

from data_quality.GE_example import check_expectations
from data_quality.create_csv_from_images import process_images_and_save_as_csv, load_images_from_folder
from data_version_controle.dvc_method import add_data_to_dvc
from deployment.deploy_model import predict_on_deployed_model, serve_model
from evaluation.check_expectations_model_deployment import model_deployment_check
from models.models import multi_unet_model
from postprocessing.visualize_predictions import visualize_segmentation_map, overlay_segmentation, \
    compare_segmentation_results
from preprocessing.preprocessing_pipeline import preprocessing
from src.train import train_and_log_model

# hyperparameters
n_classes = 3

#data_input_folder = input('Please enter the folder where the input data is stored: ')
data_input_folder = "../raw_data/AllInput"

print(f"The folder path you entered is {data_input_folder}")

#data_mask_folder = input("Please enter the folder where the labels are stored: ")
data_mask_folder = "../raw_data/AllMasks"

print(f"the folder path you entered is {data_mask_folder}")

# Save csv file on image information to use GE
input_images = process_images_and_save_as_csv(image_dir=data_input_folder, csv_filename="image_details.csv")
label_images = process_images_and_save_as_csv(image_dir=data_mask_folder, csv_filename="mask_details.csv")

# Perform Great Expectations on csv file
should_train = check_expectations(csv_file='image_details.csv', output_file='expectation_trigger_input_data.txt')

if should_train:
    #folder_path_training = input("Please enter the folder path where training input data should be stored: ")
    folder_path_training = "../data/training/input/"
    print(f"The folder path you entered is: {folder_path_training}")
    #folder_path_training_label = input("Please enter the folder path where training label data should be stored: ")
    folder_path_training_label = "../data/training/output/"
    print(f"The folder path you entered is: {folder_path_training_label}")
    #folder_path_validation = input("Please enter the folder path where validation input data should be stored: ")
    folder_path_validation = "../data/validation/input/"
    print(f"The folder path you entered is: {folder_path_validation}")
    #folder_path_validation_label = input("Please enter the folder path where validation label data should be stored: ")
    folder_path_validation_label = "../data/validation/output/"
    print(f"The folder path you entered is: {folder_path_validation_label}")
    #folder_path_testing = input("Please enter the folder path where testing input data should be stored: ")
    folder_path_testing = "../data/test/input/"
    print(f"The folder path you entered is: {folder_path_testing}")
    #folder_path_testing_label = input("Please enter the folder path where testing label data should be stored: ")
    folder_path_testing_label = "../data/test/output/"
    print(f"The folder path you entered is: {folder_path_testing_label}")

    # Preprocess images
    training_patches, training_label_patches, \
    validation_patches, validation_label_patches, \
    test_patches, test_label_patches = \
    preprocessing(input_images=input_images, label_images=label_images,
                  folder_training=folder_path_training, folder_training_label=folder_path_training_label,
                  folder_validation=folder_path_validation, folder_validation_label=folder_path_validation_label,
                  folder_testing=folder_path_testing,folder_testing_label=folder_path_testing_label,n_classes=n_classes)
    # Version the data after it has been preprocessed
    #folder_path_data = input("Please enter the folder path where all data is stored: ")
    folder_path_data = "data"
    print('Versioning data (coarse)')
    add_data_to_dvc(folder_path_data)

    # Perform training with MLFlow

    run_id, experiment_id = train_and_log_model(model=multi_unet_model(),
                                                dataset=[training_patches, training_label_patches],
                                                dataset_val=[validation_patches, validation_label_patches],
                                                n_epochs=1, n_batch=8)


'###################### NEW MODEL ########################'
experiment_id_new = '961708379921728396'
run_id_new = 'b3eac0d22bae4d9d80b7a9411bbc07d8'

# load model_old
new_model_uri = f"../src/mlartifacts/{experiment_id_new}/{run_id_new}/artifacts/mlartifacts/model"
print(f'Fetching model from: {new_model_uri}')
new_model = mlflow.tensorflow.load_model(new_model_uri)

'###################### OLD MODEL ########################'
experiment_id_old = "738313580510973847"
run_id_old = "f73227e7e38249e2a0d1f5be043c176d"

# load model_old
old_model_uri = f"../src/mlartifacts/{experiment_id_old}/{run_id_old}/artifacts/mlartifacts/model"
print(f'Fetching model from: {old_model_uri}')
old_model = mlflow.tensorflow.load_model(old_model_uri)

test_patches = load_images_from_folder(image_dir='../data/test/input')
test_label_patches = load_images_from_folder(image_dir='../data/test/output')

should_deploy = model_deployment_check(old_model=old_model, old_ex_id=experiment_id_old,
                                       old_run_id=run_id_old, new_model=new_model,
                                       new_ex_id=experiment_id_new, new_run_id=run_id_new,
                                       test_input_patches=test_patches[0:10], test_mask_patches=test_label_patches[0:10],
                                       csv_filename='deployment_check.csv', output_file='expectations_deployment.txt')

# load model
model_uri = f"../src/mlartifacts/{experiment_id_new}/{run_id_new}/artifacts/mlartifacts/model"
print(f'Fetching model from: {model_uri}')
model = mlflow.tensorflow.load_model(model_uri)

# serve model
process = serve_model(model_uri=model_uri, port=5002)

# example use of deployed model: it is available on http://127.0.0.1:port/invocations
response = predict_on_deployed_model(image_path='../data/test/input/image1split10.png', port=5002, host='127.0.0.1')

# Define your class colors
class_colors = {
    0: (0, 0, 0),        # Class 0 - Black
    1: (255, 0, 0),      # Class 1 - Blue
    2: (0, 255, 0),      # Class 2 - Green
}

if response:
    # Extract and process the prediction
    predictions = response.get('predictions', None)
    if predictions is None:
        print("No predictions found in the response.")
    else:
        prediction_array = np.array(predictions[0])
        segmentation_map = np.argmax(prediction_array, axis=-1)
        # Visualize and save the segmentation map
        visualize_segmentation_map(class_colors, segmentation_map, save_path='segmentation_map.png')

        # Overlay the segmentation on the original image and save
        overlay_segmentation(
            original_image_path='../data/test/input/image1split10.png',
            segmentation_map=segmentation_map,
            class_colors=class_colors,
            alpha=0.5,
            save_path='overlayed_image.png'
        )
        compare_segmentation_results(original_image_path='../data/test/input/image1split10.png',
                                     ground_truth_path='../data/test/input/image1split10.png',
                                     segmentation_map=segmentation_map,
                                     class_colors=class_colors,
                                     save_path='comparison.png')
else:
    print("Failed to get a valid response from the model server.")

