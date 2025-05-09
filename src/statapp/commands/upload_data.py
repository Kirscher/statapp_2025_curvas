"""
Upload data module for the statapp application.

This module provides a command to upload local directories to S3 storage.
"""

import typer
from statapp.utils.upload_utils import upload_directory_to_s3

app = typer.Typer()

@app.command()
def upload_data(
    directory: str = typer.Argument(..., help="Local directory path to upload"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
) -> None:
    """
    Upload a local directory to the S3 data folder.

    This command uploads all files and subdirectories from the specified local directory
    to the S3 data folder defined in the .env file.
    """
    upload_directory_to_s3(
        directory=directory,
        remote_dir_env_var="S3_DATA_DIR",
        verbose=verbose,
        command_description="Upload data to S3"
    )