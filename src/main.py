from data_quality.GE_example import check_expectations
from data_quality.create_csv_from_images import process_images_and_save_as_csv
from preprocessing.preprocessing_pipeline import preprocessing
from data_version_controle.dvc_method import add_data_to_dvc
from train import train_and_log_model

# Save csv file on image information to use GE
images = process_images_and_save_as_csv(image_dir="C:/PhD/Courses/MLOPS-Course/10x", csv_filename="image_details.csv")

# Perform Great Expectations on csv file
should_train = check_expectations(csv_file='C:\PhD\Courses\MLOPS-Course\data_quality\image_details.csv', output_file='expectation_trigger_input_data.txt')


if should_train:
    # Preprocess images
    patches = preprocessing(images)



# Version the data after it has been preprocessed

add_data_to_dvc('')

# Perform training with MLFlow

train_and_log_model()



print()