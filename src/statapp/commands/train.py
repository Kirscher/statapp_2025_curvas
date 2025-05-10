"""
Dataset preparation module for the statapp application.

This module provides commands for preparing datasets for analysis.
"""

import typer

from nnunetv2.run.run_training import run_training_with_args
from statapp.utils.utils import setup_logging

app = typer.Typer()

@app.command()
def train(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """
    Run nnUNet training. Must be prepared with the prepare command beforehand.

    Use --verbose to enable verbose logging output.
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Run training
    logger.info("Running nnUNet for dataset 475...")
    run_training_with_args(
        dataset_name_or_id="475",
        configuration="3d_fullres",
        fold="all",
        export_validation_probabilities=True,
        logger=logger
    )

    logger.info("nnUNet training complete!")
