import dvc.api
import os


def add_data_to_dvc(file_path):
    # Initialize DVC if not already done
    os.system('dvc init')
    # Add file to DVC
    os.system(f'dvc add {file_path}')
    # Commit the changes to DVC
    os.system('dvc commit')
    print("Changes has been committed using DVC")