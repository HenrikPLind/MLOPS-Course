import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from evaluation.evaluate_perfomance import evaluate_all_images
import great_expectations as gx


def model_deployment_check(old_model, old_ex_id, old_run_id, new_model, new_ex_id, new_run_id, test_input_patches, test_mask_patches, csv_filename):

########################## PERFORMANCE EVALUATION PART ######################################
    # Evaluate model performance
    result_old_model_class0, result_old_model_class1, result_old_model_class2\
        = (evaluate_all_images(old_model, images=test_input_patches,
                               ground_truths=test_mask_patches,
                               experiment_id=old_ex_id, run_id=old_run_id))

    # Evaluate model performance
    result_new_model_class0, result_new_model_class1, result_new_model_class2\
        = evaluate_all_images(new_model, images=test_input_patches,
                              ground_truths=test_mask_patches,
                              experiment_id=new_ex_id, run_id=new_run_id)

    performance_df = pd.DataFrame({'Dice_old_class0': result_old_model_class0,
                                   'Dice_old_class1': result_old_model_class1,
                                   'Dice_old_class2': result_old_model_class2,
                                   'Dice_new_class0': result_new_model_class0,
                                   'Dice_new_class1': result_new_model_class1,
                                   'Dice_new_class2': result_new_model_class2,
                                   })

    # Calculate avg dice for every class
    mean_old_class0 = performance_df[['Dice_old_class0']].mean()
    mean_old_class1 = performance_df[['Dice_old_class1']].mean()
    mean_old_class2 = performance_df[['Dice_old_class2']].mean()
    mean_new_class0 = performance_df[['Dice_new_class0']].mean()
    mean_new_class1 = performance_df[['Dice_new_class1']].mean()
    mean_new_class2 = performance_df[['Dice_new_class2']].mean()

    performance_mean_df = pd.DataFrame({'Dice_mean_old_class0': mean_old_class0,
                                        'Dice_mean_old_class1': mean_old_class1,
                                        'Dice_mean_old_class2': mean_old_class2,
                                        'Dice_mean_new_class0': mean_new_class0,
                                        'Dice_mean_new_class1': mean_new_class1,
                                        'Dice_mean_new_class2': mean_new_class2,})



    performance_mean_df.to_csv(csv_filename, index=False)
    print(f"CSV file saved as {csv_filename}")

########################## GREAT EXPECTATION PART ######################################
    context = gx.get_context()

    validator = context.sources.pandas_default.read_csv(csv_filename)
    # Create a Great Expectations DataFrame from the pandas DataFrame

    class_0_old_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_old_class0',
                                                                             min_value=0.01,
                                                                             max_value=1)
    class_1_old_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_old_class1',
                                                                             min_value=0.01,
                                                                             max_value=1)
    class_2_old_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_old_class2',
                                                                             min_value=0.01,
                                                                             max_value=1)
    class_0_new_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_new_class0',
                                                                             min_value=0.01,
                                                                             max_value=1)
    class_1_new_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_new_class1',
                                                                             min_value=0.01,
                                                                             max_value=1)
    class_2_new_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_new_class2',
                                                                             min_value=0.01,
                                                                             max_value=1)

    comparison_class0 = validator.expect_column_pair_values_A_to_be_greater_than_B(column_A='Dice_mean_old_class0',
                                                                                   column_B='Dice_mean_new_class0',
                                                                                   or_equal=True)
    comparison_class1 = validator.expect_column_pair_values_A_to_be_greater_than_B(column_A='Dice_mean_old_class1',
                                                                                   column_B='Dice_mean_new_class1',
                                                                                   equal=True)
    comparison_class2 = validator.expect_column_pair_values_A_to_be_greater_than_B(column_A='Dice_mean_old_class2',
                                                                                   column_B='Dice_mean_new_class2',
                                                                                   equal=True)







    columns_in_range = validator.expect_column_values_to_be_between(column='blurriness',
                                                                    min_value=0,
                                                                    max_value=30)
    image_size_expectation = validator.expect_column_pair_values_to_be_in_set(column_A='image_x_res',
                                                                              column_B='image_y_res',
                                                                              value_pairs_set=[(2708,
                                                                                                3384)])
    image_channel_expectation = validator.expect_column_distinct_values_to_equal_set(column='image_z_res',
                                                                                     value_set=([3]))






    return