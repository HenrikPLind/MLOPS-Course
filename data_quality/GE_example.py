import great_expectations as gx
import pandas as pd

def check_expectations(csv_file, output_file, alert_email=None):

    context = gx.get_context()

    validator  = context.sources.pandas_default.read_csv(csv_file)
    # Create a Great Expectations DataFrame from the pandas DataFrame

    columns_in_range = validator.expect_column_values_to_be_between(column='blurriness',
                                                                    min_value=0,
                                                                    max_value=30)
    image_size_expectation = validator.expect_column_pair_values_to_be_in_set(column_A='image_x_res',
                                                                              column_B='image_y_res',
                                                                              value_pairs_set=[(2708,
                                                                                                3384)])
    image_channel_expectation = validator.expect_column_distinct_values_to_equal_set(column='image_z_res',
                                                                                     value_set=([3]))

    # Collect results
    results = {
        "Blurriness Expectation Result": columns_in_range,
        "Image Size Expectation Result": image_size_expectation,
        "Image Channel Expectation Result": image_channel_expectation
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
        if alert_email:
            send_alert_email(alert_email, alert_message)
        return False  # Indicates that the data is not suitable for training
    else:
        return True  # Indicates that the data is suitable for training


def send_alert_email(to_email, message):
    import smtplib
    from email.mime.text import MIMEText

    # Configure your email settings
    smtp_server = "smtp.example.com" # change this to handle your own smtp server
    smtp_port = 587 # change to your port
    smtp_user = "your_email@example.com"
    smtp_password = "your_password"

    msg = MIMEText(message)
    msg['Subject'] = 'data validation Alert'
    msg['From'] = smtp_user
    msg['To'] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        print(f"Alert email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")




