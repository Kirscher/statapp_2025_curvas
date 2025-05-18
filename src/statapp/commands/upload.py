"""
Upload data, preprocessing and model artifacts module for the statapp application.

This module provides a command to upload local directories to S3 artifacts storage.
"""

import os

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

@app.command()
def upload_model_artifacts(
    directory: str = typer.Argument(..., help="Local directory path to upload"),
    modelfolder: str = typer.Argument(..., help="Subfolder name of the model"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
) -> None:
    """
    Upload a local directory to the S3 artifacts/model folder.

    This command uploads all files and subdirectories from the specified local directory
    to the S3 artifacts folder defined in the .env file. If a subfolder is specified,
    the files will be placed in that subfolder within the artifacts/models directory.
    """
    # Construct the subfolder path: models/modelfolder
    model_subfolder = f"{os.environ.get('S3_MODEL_ARTIFACTS_SUBDIR', 'models')}/{modelfolder}"

    upload_directory_to_s3(
        directory=directory,
        remote_dir_env_var="S3_ARTIFACTS_DIR",
        subfolder=model_subfolder,
        verbose=verbose,
        command_description="Upload model artifacts to S3"
    )

@app.command()
def upload_preprocessing_artifacts(
    directory: str = typer.Argument(..., help="Local directory path to upload"),
    preprocessingfolder: str = typer.Argument(..., help="Subfolder name of the preprocessing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
) -> None:
    """
    Upload a local directory to the S3 artifacts/preprocessing folder.

    This command uploads all files and subdirectories from the specified local directory
    to the S3 artifacts folder defined in the .env file. If a subfolder is specified,
    the files will be placed in that subfolder within the artifacts/preprocessing directory.
    """
    # Construct the subfolder path: preprocessing/preprocessingfolder
    preprocessing_subfolder = f"{os.environ.get('S3_PROPROCESSING_ARTIFACTS_SUBDIR', 'preprocessing')}/{preprocessingfolder}"

    upload_directory_to_s3(
        directory=directory,
        remote_dir_env_var="S3_ARTIFACTS_DIR",
        subfolder=preprocessing_subfolder,
        verbose=verbose,
        command_description="Upload preprocessing artifacts to S3"
    )
