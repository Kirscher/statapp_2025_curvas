"""
Empty data module for the statapp application.

This module provides a command to remove all files and folders from the data directory in S3 storage.
"""

import os

import typer
from rich.text import Text

from statapp.utils import s3_utils
from statapp.utils import utils

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
    # Setup logging
    logger = utils.setup_logging(verbose)

    # Get data directory path
    data_dir = f"{os.environ['S3_BUCKET']}/{os.environ['S3_DATA_DIR']}"

    # Display warning
    warning_text = Text.assemble(
        ("WARNING: ", "bold red"),
        ("This will delete ", "bold"),
        ("ALL", "bold red"),
        (" files and folders in ", "bold"),
        (data_dir, "bold red"),
        (".\nThis operation cannot be undone!", "bold")
    )
    utils.info(warning_text)

    # Confirm deletion if not already confirmed
    if not confirm:
        confirmation = typer.confirm("Are you sure you want to proceed?", default=False)
        if not confirmation:
            utils.info(Text("Operation cancelled.", style="bold yellow"))
            return

    # List files before deletion
    objects = s3_utils.list_data_directory()
    num_objects = len(objects)

    if num_objects == 0:
        utils.info(Text(f"The data directory {data_dir} is already empty.", style="bold green"))
        return

    # Display info
    info_text = Text.assemble(
        ("Deleting ", "bold"),
        (str(num_objects), "bold red"),
        (" objects from ", "bold"),
        (data_dir, "bold red")
    )
    utils.info(info_text)

    # Delete all objects
    try:
        deleted_objects = s3_utils.empty_data_directory()

        # Display completion message
        success_text = Text.assemble(
            ("Data directory emptied successfully!", "bold green"),
            ("\nDeleted ", "bold"),
            (str(len(deleted_objects)), "bold green"),
            (" objects from ", "bold"),
            (data_dir, "bold green")
        )
        utils.info(success_text)

        # Log deleted objects if verbose
        if verbose:
            logger.info(f"Deleted objects: {', '.join(deleted_objects)}")

    except Exception as e:
        error_text = Text(f"Error emptying data directory: {str(e)}", style="bold red")
        utils.info(error_text)
        logger.error(f"Error emptying data directory: {str(e)}")