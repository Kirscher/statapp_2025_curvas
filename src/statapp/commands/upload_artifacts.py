"""
Upload artifacts module for the statapp application.

This module provides a command to upload local directories to S3 artifacts storage.
"""

import typer
from statapp.utils.upload_utils import upload_directory_to_s3

app = typer.Typer()

@app.command()
def upload_artifacts(
    directory: str = typer.Argument(..., help="Local directory path to upload"),
    subfolder: str = typer.Argument(..., help="Subfolder name within the artifacts directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
) -> None:
    """
    Upload a local directory to the S3 artifacts folder.

    This command uploads all files and subdirectories from the specified local directory
    to the S3 artifacts folder defined in the .env file. If a subfolder is specified,
    the files will be placed in that subfolder within the artifacts directory.
    """
    upload_directory_to_s3(
        directory=directory,
        remote_dir_env_var="S3_ARTIFACTS_DIR",
        subfolder=subfolder,
        verbose=verbose,
        command_description="Upload artifacts to S3"
    )
