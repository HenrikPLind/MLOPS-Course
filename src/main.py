from data_quality.GE_example import check_expectations
from data_quality.create_csv_from_images import process_images_and_save_as_csv
from data_version_controle.dvc_method import add_data_to_dvc
from deployment.deploy_model import deploy_with_mlflow
from evaluation.evaluate_perfomance import evaluate_segmentation
from models.models import multi_unet_model
from preprocessing.preprocessing_pipeline import preprocessing
from train import train_and_log_model

#data_input_folder = input('Please enter the folder where the input data is stored: ')
data_input_folder = "D:/MLOPS/Data/AllInput"
print(f"The folder path you entered is {data_input_folder}")

#data_mask_folder = input("Please enter the folder where the labels are stored: ")
data_mask_folder = "D:/MLOPS/Data/AllMasks"
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
                  folder_testing=folder_path_testing,folder_testing_label=folder_path_testing_label)
    # Version the data after it has been preprocessed
    #folder_path_data = input("Please enter the folder path where all data is stored: ")
    folder_path_data = "data"
    print('Versioning data (coarse)')
    add_data_to_dvc(folder_path_data)


    # Perform training with MLFlow
    run_id, experiment_id = train_and_log_model(model=multi_unet_model(), dataset=[training_patches, training_label_patches],
                            dataset_val=[validation_patches, validation_label_patches],
                            label_mask_path_train=folder_path_training_label,
                            label_mask_path_val=folder_path_validation_label,
                            model_name='multi_unet', n_epochs=1, n_batch=8)

    # Evaluate model performance
    result = evaluate_segmentation(image=test_patches[0], ground_truth=test_label_patches[0],
                                   experiment_id=experiment_id, run_id=run_id)

    print()


    # Deploy model
    deploy_with_mlflow()











print()