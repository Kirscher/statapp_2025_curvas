"""
Empty data module for the statapp application.

This module provides a command to remove all files and folders from the data directory in S3 storage.
"""

import typer

from statapp.utils import s3_utils
from statapp.utils.empty_utils import empty_directory

app = typer.Typer()

@app.command()
def empty_data(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    confirm: bool = typer.Option(False, "--confirm", "-c", help="Confirm deletion without prompting")
) -> None:
    """
    Remove all files and folders from the S3 data folder.

    This command deletes all objects in the S3 data folder defined in the .env file.
    Use with caution as this operation cannot be undone.
    """
    empty_directory(
        directory_env_var="S3_DATA_DIR",
        list_directory_func=s3_utils.list_data_directory,
        empty_directory_func=s3_utils.empty_data_directory,
        verbose=verbose,
        confirm=confirm,
        command_description="Empty data directory"
    )
