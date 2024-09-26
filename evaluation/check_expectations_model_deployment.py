import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from evaluation.evaluate_perfomance import evaluate_all_images
import great_expectations as gx


def model_deployment_check(old_model, old_ex_id, old_run_id, new_model, new_ex_id,
                           new_run_id, test_input_patches, test_mask_patches,
                           csv_filename, output_file):

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
    mean_old_class0 = performance_df['Dice_old_class0'].mean()
    mean_old_class1 = performance_df['Dice_old_class1'].mean()
    mean_old_class2 = performance_df['Dice_old_class2'].mean()
    mean_new_class0 = performance_df['Dice_new_class0'].mean()
    mean_new_class1 = performance_df['Dice_new_class1'].mean()
    mean_new_class2 = performance_df['Dice_new_class2'].mean()

    performance_mean_df = pd.DataFrame({'Dice_mean_old_class0': [mean_old_class0],
                                        'Dice_mean_old_class1': [mean_old_class1],
                                        'Dice_mean_old_class2': [mean_old_class2],
                                        'Dice_mean_new_class0': [mean_new_class0],
                                        'Dice_mean_new_class1': [mean_new_class1],
                                        'Dice_mean_new_class2': [mean_new_class2],})



    performance_mean_df.to_csv(csv_filename, index=False)
    print(f"CSV file saved as {csv_filename}")

########################## GREAT EXPECTATION PART ######################################
    context = gx.get_context()

    validator = context.sources.pandas_default.read_csv(csv_filename)
    # Create a Great Expectations DataFrame from the pandas DataFrame

    class_0_old_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_old_class0',
                                                                             min_value=0.00001,
                                                                             max_value=1)
    class_1_old_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_old_class1',
                                                                             min_value=0.00001,
                                                                             max_value=1)
    class_2_old_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_old_class2',
                                                                             min_value=0.00001,
                                                                             max_value=1)
    class_0_new_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_new_class0',
                                                                             min_value=0.00001,
                                                                             max_value=1)
    class_1_new_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_new_class1',
                                                                             min_value=0.00001,
                                                                             max_value=1)
    class_2_new_above_threshold = validator.expect_column_min_to_be_between(column='Dice_mean_new_class2',
                                                                             min_value=0.00001,
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


# Collect results
    results = {
        "Class 0 old threshold": class_0_old_above_threshold,
        "Class 1 old threshold": class_1_old_above_threshold,
        "Class 2 old threshold": class_2_old_above_threshold,
        "Class 0 new threshold": class_0_new_above_threshold,
        "Class 1 new threshold": class_1_new_above_threshold,
        "Class 2 new threshold": class_2_new_above_threshold,
        "Comparison class 0": comparison_class0,
        "Comparison class 1": comparison_class1,
        "Comparison class 2": comparison_class2
    }

# Write results to a file
    with open(output_file, 'w') as f:
        for key, result in results.items():
            f.write(f"{key}:\n")
            f.write(str(result) + "\n\n")

    print(f"Results saved to {output_file}")

    # Check and alert if any expectations are not met
    failed_expectations = [key for key, result in results.items() if not result['success']]

    if failed_expectations:
        alert_message = f"Expectations failed for the following checks: {', '.join(failed_expectations)}"
        print(alert_message)
        return False  # Indicates that the data is not suitable for training
    else:
        return True  # Indicates that the data is suitable for training
