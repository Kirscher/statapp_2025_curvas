"""
Upload utilities module for the statapp application.

This module provides shared functionality for uploading files to S3 storage.
"""

import os
import time
from pathlib import Path

from rich.text import Text

import statapp.utils as utils
from statapp.utils import s3_utils
from statapp.utils.progress_tracker import ProgressTracker, track_progress


def upload_directory_to_s3(
    directory: str,
    remote_dir_env_var: str,
    verbose: bool = False,
    command_description: str = "Upload files to S3"
) -> None:
    """
    Upload a local directory to an S3 folder.

    Args:
        directory (str): Local directory path to upload
        remote_dir_env_var (str): Environment variable name for the remote directory
        verbose (bool): Enable verbose output
        command_description (str): Description of the command for logging
    """
    # Setup logging
    logger = utils.setup_logging(verbose)

    # Convert to absolute path if it's a relative path
    directory_path = Path(directory)
    if not directory_path.is_absolute():
        directory_path = Path.cwd() / directory_path

    # Check if directory exists
    if not directory_path.exists() or not directory_path.is_dir():
        error_text = Text(f"Error: Directory '{directory_path}' does not exist or is not a directory", style="bold red")
        utils.info(error_text)
        return

    # Prepare remote path
    remote_base_path = f"{os.environ['S3_BUCKET']}/{os.environ[remote_dir_env_var]}"

    # Display info
    info_text = Text.assemble(
        ("Uploading directory: ", "bold"),
        (str(directory_path), "bold green"),
        ("\nTo S3 path: ", "bold"),
        (remote_base_path, "bold green")
    )
    utils.info(info_text)

    # Get all files in the directory and subdirectories
    all_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            local_file_path = Path(root) / file
            all_files.append(local_file_path)

    # Define a function to get the size of a file
    def get_file_size(file_path: Path) -> int:
        return file_path.stat().st_size

    # Define a function to process a file
    def process_file(local_file_path: Path, tracker: ProgressTracker) -> None:
        # Calculate relative path from the base directory
        rel_path = local_file_path.relative_to(directory_path)
        file_size = local_file_path.stat().st_size

        # Check if the file name itself contains backslashes (indicating a folder structure)
        file_name = local_file_path.name
        if '\\' in file_name:
            # Split the file name into folder and actual file name
            folder, actual_file = file_name.split('\\', 1)
            # Create a new path with the folder structure
            rel_path = Path(rel_path.parent) / folder / actual_file

        # Truncate path if too long
        display_path = str(rel_path)
        if len(display_path) > 30:
            display_path = "..." + display_path[-27:]

        # Construct remote path - replace Windows backslashes with forward slashes
        remote_path = f"{remote_base_path}/{str(rel_path).replace('\\', '/')}"

        # Log the upload (with simplified path)
        logger.info(f"Uploading {display_path} to S3")

        try:
            # Start tracking progress for this file
            tracker.start_file(local_file_path, display_path, file_size)

            # Record start time
            start_time = time.time()

            # Create a callback function for progress updates that's compatible with boto3
            def update_progress(bytes_transferred: int) -> None:
                tracker.update_file_progress(bytes_transferred, file_size, display_path, start_time)

            # Upload the file with progress tracking using boto3
            s3_utils.upload_file(str(local_file_path), remote_path, callback=update_progress)

            # Mark file as completed
            tracker.complete_file(display_path, file_size, start_time)

        except Exception as e:
            logger.error(f"Error uploading {local_file_path}: {str(e)}")
            tracker.complete_file(display_path, file_size, time.time(), success=False)

    # Track progress of uploading files
    track_progress(all_files, get_file_size, process_file)

    # Display completion message
    success_text = Text.assemble(
        ("Upload completed!", "bold green"),
        ("\nUploaded ", "bold"),
        (str(len(all_files)), "bold green"),
        (" files to ", "bold"),
        (remote_base_path, "bold green")
    )
    utils.info(success_text)