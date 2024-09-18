import subprocess
import os


def add_data_to_dvc(file_path, auto_commit=False):
    # Determine the parent directory where .dvc should be located (one level up)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Check if DVC has already been initialized in the parent directory
    dvc_path = os.path.join(parent_dir, '.dvc')
    if not os.path.exists(dvc_path):
        print(f"Initializing DVC in the repository at {parent_dir}...")
        subprocess.run(['dvc', 'init'], cwd=parent_dir, check=True)

    # Add file to DVC (ensure we are adding it relative to the correct directory)
    print(f"Adding {file_path} to DVC in {parent_dir}...")
    subprocess.run(['dvc', 'add', os.path.join(parent_dir, file_path)], cwd=parent_dir, check=True)

    # Inform the user that the file is now tracked by DVC
    print(f"File '{file_path}' has been added to DVC.")

    # Auto-commit to Git if the option is enabled
    if auto_commit:
        print(f"Committing DVC changes for {file_path} to Git...")

        # Add .dvc file and .gitignore to Git
        subprocess.run(['git', 'add', f'{file_path}.dvc', '.gitignore'], cwd=parent_dir, check=True)

        # Commit changes
        commit_message = f"Add {file_path} to DVC"
        subprocess.run(['git', 'commit', '-m', commit_message], cwd=parent_dir, check=True)

        print("Changes have been committed to Git.")
    else:
        # Reminder to commit to Git manually
        print("Please remember to commit the changes using Git:")
        print(f"git add {file_path}.dvc .gitignore")
        print("git commit -m 'Added file to DVC'")