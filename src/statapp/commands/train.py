"""
Dataset preparation module for the statapp application.

This module provides commands for preparing datasets for analysis.
"""

import os
import typer
from typing import Optional

from nnunetv2.run.run_training import run_training_with_args
from statapp.utils.utils import setup_logging, info
from rich.text import Text

app = typer.Typer()

@app.command()
def train(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Set random seed for reproducibility"),
) -> None:
    """
    Run nnUNet training. Must be prepared with the prepare command beforehand.

    Use --verbose to enable verbose logging output.
    Use --seed to set a random seed for reproducible results.
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Set seed environment variable if provided
    if seed is not None:
        os.environ['SEED'] = str(seed)
        info(Text.assemble(
            ("Random seed set to: ", "bold green"),
            (f"{seed}", "bold cyan")
        ))

    # Run training
    info(Text("Running nnUNet for dataset 475...", style="bold blue"))
    run_training_with_args(
        dataset_name_or_id="475",
        configuration="3d_fullres",
        trainer_name="nnUNetTrainer_Statapp",
        fold="all",
        export_validation_probabilities=True,
        logger=logger
    )

    info(Text("nnUNet training complete!", style="bold green"))
