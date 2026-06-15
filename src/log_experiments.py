"""
this module saves the code at each experiment that we run
to not lose track of small changes between the commits
"""
import zipfile
import os


def save_code(zip_file_name, folder="."):
    """
    Creates a zip archive of the specified folder.

    Parameters:
    - zip_file_name: str, name of the output zip file
    - folder: str, path to the folder to zip (default: ".")

    Returns:
    - None
    """
    os.makedirs(os.path.dirname(zip_file_name), exist_ok=True)
    files_to_zip = [f for f in os.listdir(folder) if ".py" in f]

    with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            if os.path.exists(file_path):
                # Add the file to the zip archive
                zipf.write(file_path, os.path.basename(file_path))
            else:
                print(f"Warning: File not found - {file_path}")

    print(f"Successfully created {zip_file_name}.")
