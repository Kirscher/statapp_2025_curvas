"""
Empty utilities module for the statapp application.

This module provides shared functionality for emptying directories in S3 storage.
"""

import os
from typing import Callable, List, Dict, Any

import typer
from rich.text import Text

from statapp.utils import utils


def empty_directory(
    directory_env_var: str,
    list_directory_func: Callable[[], List[Dict[str, Any]]],
    empty_directory_func: Callable[[], List[str]],
    verbose: bool = False,
    confirm: bool = False,
    command_description: str = "Empty directory"
) -> None:
    """
    Empty a directory in S3 storage.

    Args:
        directory_env_var (str): Environment variable name for the directory
        list_directory_func (callable): Function to list contents of the directory
        empty_directory_func (callable): Function to empty the directory
        verbose (bool): Enable verbose output
        confirm (bool): Confirm deletion without prompting
        command_description (str): Description of the command for logging
    """
    # Setup logging
    logger = utils.setup_logging(verbose)

    # Extract directory type from command description
    directory_type = command_description.split()[-2] if len(command_description.split()) > 1 else "directory"

    # Get directory path
    directory_path = f"{os.environ['S3_BUCKET']}/{os.environ[directory_env_var]}"

    # Display warning
    warning_text = Text.assemble(
        ("WARNING: ", "bold red"),
        ("This will delete ", "bold"),
        ("ALL", "bold red"),
        (" files and folders in ", "bold"),
        (directory_path, "bold red"),
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
    objects = list_directory_func()
    num_objects = len(objects)

    if num_objects == 0:
        utils.info(Text(f"The {directory_type} directory {directory_path} is already empty.", style="bold green"))
        return

    # Display info
    info_text = Text.assemble(
        ("Deleting ", "bold"),
        (str(num_objects), "bold red"),
        (" objects from ", "bold"),
        (directory_path, "bold red")
    )
    utils.info(info_text)

    # Delete all objects
    try:
        deleted_objects = empty_directory_func()

        # Display completion message
        success_text = Text.assemble(
            (f"{directory_type.capitalize()} directory emptied successfully!", "bold green"),
            ("\nDeleted ", "bold"),
            (str(len(deleted_objects)), "bold green"),
            (" objects from ", "bold"),
            (directory_path, "bold green")
        )
        utils.info(success_text)

        # Log deleted objects if verbose
        if verbose:
            logger.info(f"Deleted objects: {', '.join(deleted_objects)}")

    except Exception as e:
        error_text = Text(f"Error emptying {directory_type} directory: {str(e)}", style="bold red")
        utils.info(error_text)
        logger.error(f"Error emptying {directory_type} directory: {str(e)}")
