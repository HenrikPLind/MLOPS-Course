from data_quality.GE_example import check_expectations
from data_quality.create_csv_from_images import process_images_and_save_as_csv
from preprocessing.preprocessing_pipeline import preprocessing

# Save csv file on image informaiton to use GE
images = process_images_and_save_as_csv(image_dir="C:/PhD/Courses/MLOPS-Course/10x", csv_filename="image_details.csv")

# Perform Great Expectations on csv file
check_expectations(csv_file='C:\PhD\Courses\MLOPS-Course\data_quality\image_details.csv', output_file='expectation_trigger_input_data.txt')

# Preprocess images
patches = preprocessing(images)



print()