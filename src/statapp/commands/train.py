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
    seed: Optional[int] = typer.Argument(None, help="Set random seed for reproducibility"),
    fold: Optional[str] = typer.Option("all", "--fold", "-f", help="Fold to use for training. Can be 'all' to use all folds, or a specific fold number (0-4)."),

) -> None:
    """
    Run nnUNet training. Must be prepared with the prepare command beforehand.

    Use --verbose to enable verbose logging output.
    Use --seed to set a random seed for reproducible results.
    Use --fold to specify which fold to use for training (0-4) or 'all' to use all folds.
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Run training
    info(Text("Running nnUNet training...", style="bold blue"))
    # Set seed environment variable if provided
    if seed is not None:
        os.environ['SEED'] = str(seed)
        logger.info(Text.assemble(
            ("Random seed set to: ", "bold green"),
            (f"{seed}", "bold cyan")
        ))

    # Validate fold parameter
    if fold != "all":
        try:
            fold_num = int(fold)
            if fold_num < 0 or fold_num > 4:
                raise ValueError(f"Fold number must be between 0 and 4, got {fold_num}")
            logger.info(Text.assemble(
                ("Using fold: ", "bold green"),
                (f"{fold_num}", "bold cyan")
            ))
        except ValueError:
            error_msg = f"Invalid fold value: {fold}. Must be 'all' or an integer between 0 and 4."
            logger.error(Text(error_msg, style="bold red"))
            raise typer.BadParameter(error_msg)
    else:
        logger.info(Text("Using all folds for training", style="bold green"))

    run_training_with_args(
        dataset_name_or_id="475",
        configuration="3d_fullres",
        trainer_name="nnUNetTrainer_Statapp",
        fold=fold,
        export_validation_probabilities=False, #too big for Onyxia
        logger=logger
    )

    info(Text("nnUNet training complete!", style="bold green"))
