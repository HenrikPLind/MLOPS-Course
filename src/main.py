from data_quality.GE_example import check_expectations
from data_quality.create_csv_from_images import process_images_and_save_as_csv
from preprocessing.preprocessing_pipeline import preprocessing
from data_version_controle.dvc_method import add_data_to_dvc
#from train import train_and_log_model

data_input_folder = input('Please enter the folder where the input data is stored: ')
print(f"The folder path you entered is {data_input_folder}")

data_mask_folder = input("Please enter the folder where the labels are stored: ")
print(f"the folder path you entered is {data_mask_folder}")

# Save csv file on image information to use GE
input_images = process_images_and_save_as_csv(image_dir=data_input_folder, csv_filename="image_details.csv")
label_images = process_images_and_save_as_csv(image_dir=data_mask_folder, csv_filename="mask_details.csv")

# Perform Great Expectations on csv file
should_train = check_expectations(csv_file='image_details.csv', output_file='expectation_trigger_input_data.txt')

if should_train:
    folder_path_training = input("Please enter the folder path where training data should be stored: ")
    print(f"The folder path you entered is: {folder_path_training}")
    folder_path_validation = input("Please enter the folder path where training data should be stored: ")
    print(f"The folder path you entered is: {folder_path_validation}")
    folder_path_test = input("Please enter the folder path where training data should be stored: ")
    print(f"The folder path you entered is: {folder_path_test}")
    # Preprocess images
    patches = preprocessing(input_images=input_images, label_images=label_images,
                            folder_training=folder_path_training,
                            folder_validation=folder_path_validation,
                            folder_testing=folder_path_test)
    # Version the data after it has been preprocessed
    print('Versioning data')
    add_data_to_dvc(folder_path)
    # Perform training with MLFlow
    train_and_log_model()








print()