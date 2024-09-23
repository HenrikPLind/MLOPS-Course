import mlflow.keras
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from evaluation.evaluate_perfomance import evaluate_all_images
from src.Helpers.patch_loader import load_patches

#hyperparameters
n_classes = 3

#data_input_folder = input('Please enter the folder where the input data is stored: ')
#data_input_folder = "D:/MLOPS/Data/AllInput"
data_input_folder = "C:/Users/mose_/Desktop/dataHenrik/AllInput - Copy"

print(f"The folder path you entered is {data_input_folder}")

#data_mask_folder = input("Please enter the folder where the labels are stored: ")
#data_mask_folder = "D:/MLOPS/Data/AllMasks"
data_mask_folder = "C:/Users/mose_/Desktop/dataHenrik/AllInput - Copy"

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

experiment_id = "568984660945487376"
run_id = "30cf9d6599a4418886b27ca507a884b8"

# load model
model_uri = f"../src/mlartifacts/{experiment_id}/{run_id}/artifacts/mlartifacts/model"
print(f'Fetching model from: {model_uri}')
model = mlflow.tensorflow.load_model(model_uri)

test_patches = load_patches(
    "C:/Users/mose_/AAU/Sundhedstek/PhD/Courses/Data and Machine Learning Operations/Code_mlops_Course/MLOPS-Course/data/test/input/")
test_label_patches = load_patches(
    "C:/Users/mose_/AAU/Sundhedstek/PhD/Courses/Data and Machine Learning Operations/Code_mlops_Course/MLOPS-Course/data/test/output/")

label_images = np.array(test_label_patches)
label_images = np.expand_dims(label_images, axis=3)
label_images = to_categorical(label_images, num_classes=3)
label_images = label_images.reshape((label_images.shape[0], label_images.shape[1], label_images.shape[2], 3))
label_images = label_images.astype(np.uint8)
label_images = list(label_images)


# Evaluate model performance
result = evaluate_all_images(model, images=test_patches, ground_truths=test_label_patches,
                             experiment_id=experiment_id, run_id=run_id)

bob = 0