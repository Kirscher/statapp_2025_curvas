"""
Utilities for the statapp CLI application.

This module provides utility functions for display and logging.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.text import Text

# Create a shared console instance for consistent output
console = Console()


name = "ENSAE 2025 applied statistics project"
version = "Version 1.0.0"

def info(text: Text) -> None:
    """
    Display an information message in a styled panel.

    Args:
        text (Text): The text to display
    """
    panel = Panel(text, title=name, title_align="left", subtitle=f"{version}", subtitle_align="left")
    console.print(panel)
    return None

def pretty_print(text: str) -> None:
    """
    Display text with rich formatting.

    Args:
        text (str): The text to display
    """
    console.print(text)
    return None

def create_progress(description: str = "Processing") -> Progress:
    """
    Create and return a customized Progress object.

    Args:
        description (str, optional): Description for the progress bar. Defaults to "Processing".

    Returns:
        Progress: The configured Progress object
    """
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
        "•",
        TextColumn("[bold cyan]{task.fields[file_size]}", justify="right"),
        "•",
        TextColumn("[bold green]{task.fields[speed]}", justify="right"),
        refresh_per_second=10,  # More frequent updates
        transient=True,  # Use the same line for updates
        console=console  # Use the shared console instance
    )

    # Add a task with the provided description
    progress.add_task(description, total=100, file_size="0 B", speed="0 B/s")

    return progress

def create_dual_progress() -> Progress:
    """
    Create and return a customized Progress object with two progress bars.

    One for overall progress and one for per-file progress.

    Returns:
        Progress: The configured Progress object with two progress bars
    """
    return Progress(
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
        "•",
        TextColumn("{task.fields[size]}", justify="right"),
        "•",
        TextColumn("{task.fields[elapsed]}", justify="right"),
        refresh_per_second=10,  # More frequent updates
        expand=True,  # Expand to full width
        transient=True,  # Use transient to update in place
        console=console  # Use the shared console instance
    )

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure and return a logger for the application.

    Args:
        verbose (bool, optional): If True, display debug messages in the console. Defaults to False.

    Returns:
        logging.Logger: The configured logger
    """
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_folder, f"log_{timestamp}.log")

    # Remove all handlers for root logger
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)

    # File handler (DEBUG level) with UTF-8 encoding
    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Rich console handler (INFO or DEBUG level based on verbose)
    # This integrates with the shared console instance
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=False,
        show_path=False,
        level=logging.DEBUG if verbose else logging.INFO
    )
    logger.addHandler(rich_handler)

    return logger


def setup_nnunet_env(base: Path, raw: Path, preprocessed: Path, results: Path) -> None:
    """Sets up variable environnement to configure nnunet to use a given path

    Args:
        base (Path): base path of the data folder
        raw (Path): path relative to base of the raw image folder
        preprocessed (Path): path relative to base of the preprocessed image folder
        results (Path): path relative to base of the results

    """
    env_vars = {
        'nnUNet_raw': str(base / raw),
        'nnUNet_preprocessed': str(base / preprocessed),
        'nnUNet_results': str(base / results)
    }

    for var_name, path in env_vars.items():
        os.environ[var_name] = path

    return
