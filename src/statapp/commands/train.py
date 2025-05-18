"""
Dataset preparation module for the statapp application.

This module provides commands for preparing datasets for analysis.
"""

import os
from pathlib import Path
from typing import Optional, List

import typer
from rich.text import Text

from nnunetv2.run.run_training import run_training_with_args
from statapp.commands.prepare import get_dataset_code, DATASET_PREFIX, download_preprocessing
from statapp.utils.utils import setup_logging, info

app = typer.Typer()

@app.command()
def train(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    seed: Optional[int] = typer.Argument(None, help="Set random seed for reproducibility"),
    fold: Optional[str] = typer.Option("all", "--fold", "-f", help="Fold to use for training. Can be 'all' to use all folds, or a specific fold number (0-4)."),
    patients: List[str] = typer.Option(["all"], "--patients", "-p", help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    annotator: str = typer.Option("1", "--annotator", "-a", help="Annotator (1/2/3)"),
) -> None:
    """
    Run nnUNet training. Must be prepared with the prepare command beforehand.

    Use --verbose to enable verbose logging output.
    Use --seed to set a random seed for reproducible results.
    Use --fold to specify which fold to use for training (0-4) or 'all' to use all folds.
    Use --patients to specify which patient set to use ('all', 'train', 'validation', 'test', or a custom list).
    Use --annotator to specify which annotator's data to use (1/2/3).
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

    # Validate annotator input
    if annotator not in ["1", "2", "3"]:
        error_msg = f"Annotator must be 1, 2, or 3. Got {annotator}"
        logger.error(Text(error_msg, style="bold red"))
        raise typer.BadParameter(error_msg)

    # Handle patient selection
    patient_selection = patients
    if len(patients) == 1:
        # Check if it's one of the predefined sets
        if patients[0] in ["all", "train", "validation", "test"]:
            patient_selection = patients[0]
            logger.info(Text.assemble(
                ("Using patient set: ", "bold green"),
                (f"{patient_selection}", "bold cyan")
            ))
    else:
        logger.info(Text.assemble(
            ("Using custom patient list: ", "bold green"),
            (f"{', '.join(patient_selection)}", "bold cyan")
        ))

    # Generate dataset code for folder naming
    dataset_code = get_dataset_code(patient_selection)

    # Construct the dataset ID with annotator and dataset code
    dataset_id = f"{DATASET_PREFIX}{annotator}_{dataset_code}"
    logger.info(Text.assemble(
        ("Using dataset: ", "bold green"),
        (f"{dataset_id}", "bold cyan")
    ))

    # Check if preprocessing artifacts exist locally
    preprocessing_path = Path(f"nnUNet_preprocessed/{dataset_id}")
    plans_file = preprocessing_path / "nnUNetPlans.json"

    if not plans_file.exists():
        logger.info(f"Preprocessing artifacts not found locally at {preprocessing_path}")
        logger.info("Attempting to download preprocessing artifacts from S3...")

        # Try to download preprocessing artifacts from S3
        preprocessing_exists = download_preprocessing(
            annotator=annotator,
            dataset_code=dataset_code,
            verbose=verbose
        )

        if not preprocessing_exists:
            logger.error(Text("Preprocessing artifacts not found in S3. Please run the prepare command first.", style="bold red"))
            return

        # Verify that the download was successful
        if not plans_file.exists():
            logger.error(Text("Failed to download preprocessing artifacts. Please run the prepare command first.", style="bold red"))
            return

        logger.info(Text("Successfully downloaded preprocessing artifacts from S3", style="bold green"))
    else:
        logger.info(Text(f"Using existing preprocessing artifacts at {preprocessing_path}", style="bold green"))

    run_training_with_args(
        dataset_name_or_id=dataset_id,
        configuration="3d_fullres",
        trainer_name="nnUNetTrainer_Statapp",
        fold=fold,
        export_validation_probabilities=False, #too big for Onyxia
        logger=logger
    )

    info(Text("nnUNet training complete!", style="bold green"))
