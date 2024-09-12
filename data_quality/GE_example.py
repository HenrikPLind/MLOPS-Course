import great_expectations as gx
import pandas as pd

def check_expectations(csv_file, output_file):

    context = gx.get_context()

    validator  = context.sources.pandas_default.read_csv(csv_file)
    # Create a Great Expectations DataFrame from the pandas DataFrame

    columns_in_range = validator.expect_column_values_to_be_between(column='blurriness', min_value=0, max_value=15)
    image_size_expectation = validator.expect_column_pair_values_to_be_in_set(column_A='image_x_res', column_B='image_y_res', value_pairs_set=[(2708, 3384)])
    image_channel_expectation = validator.expect_column_distinct_values_to_equal_set(column='image_z_res', value_set=([2]))

    # Open a file to save the results
    with open(output_file, 'w') as f:
        f.write("Blurriness Expectation Result:\n")
        f.write(str(columns_in_range) + "\n\n")

        f.write("Image Size Expectation Result:\n")
        f.write(str(image_size_expectation) + "\n\n")

        f.write("Image Channel Expectation Result:\n")
        f.write(str(image_channel_expectation) + "\n\n")

    print(f"Results saved to {output_file}")




