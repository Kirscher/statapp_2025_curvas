"""
All-in-one command module for the statapp application.

This module provides a command that combines prepare, train, and upload_artifacts functionality.
"""

import os
import typer
from typing import Optional, List

from statapp.commands.prepare import prepare, get_dataset_code
from statapp.commands.train import train
from statapp.commands.upload import upload_model_artifacts
from statapp.utils.utils import setup_logging, info
from rich.text import Text

app = typer.Typer()

@app.command()
def run(
    annotator: str = typer.Argument(..., help="Annotator (1/2/3)"),
    seed: int = typer.Argument(..., help="Set random seed for reproducibility"),
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    fold: str = typer.Option("all", "--fold", "-f", help="Fold to use for training. Can be 'all' to use all folds, or a specific fold number (0-4)."),
    skip: bool = typer.Option(False, "--skip", help="Skip download and only run preprocessing"),
    num_processes_fingerprint: int = typer.Option(2, "--num-processes-fingerprint", "-npfp", help="Number of processes to use for fingerprint extraction"),
    num_processes: int = typer.Option(2, "--num-processes", "-np", help="Number of processes to use for preprocessing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
) -> None:
    """
    Run the complete pipeline: prepare data, train model, and upload artifacts.

    This command combines the functionality of prepare, train, and upload_artifacts commands.
    It first prepares the data, then trains the model, and finally uploads the artifacts.

    Patient selection options:
    - 'all': All available patients
    - 'train': Predefined training set
    - 'validation': Predefined validation set
    - 'test': Predefined test set
    - Custom list: Specific patient numbers (e.g., 001 034)

    Use --verbose to enable verbose logging output.
    Use --skip to skip download and only run preprocessing.
    Use --fold to specify which fold to use for training (0-4) or 'all' to use all folds.
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Handle patient selection
    patient_selection = patients
    if len(patients) == 1:
        # Check if it's one of the predefined sets
        if patients[0] in ["all", "train", "validation", "test"]:
            patient_selection = patients[0]

    # Generate dataset code for folder naming
    dataset_code = get_dataset_code(patient_selection)

    # Step 1: Run prepare command
    info(Text("Step 1: Preparing data...", style="bold blue"))
    prepare(
        annotator=annotator,
        patients=patients,
        skip=skip,
        num_processes_fingerprint=num_processes_fingerprint,
        num_processes=num_processes,
        verbose=verbose
    )

    # Step 2: Run train command
    info(Text("Step 2: Training model...", style="bold blue"))
    # Set seed environment variable
    os.environ['SEED'] = str(seed)
    train(
        verbose=verbose,
        seed=seed,
        fold=fold,
        patients=patients,
        annotator=annotator
    )

    # Step 3: Upload artifacts
    info(Text("Step 3: Uploading artifacts...", style="bold blue"))

    # Construct the directory path based on the requirements
    directory = f"nnUnet_results/Dataset475_CURVAS_ANNO{annotator}_{dataset_code}"

    # Construct the subfolder name based on the requirements
    subfolder = f"anno{annotator}_init{seed}_fold{fold}"

    # Upload the model artifacts
    upload_model_artifacts(
        directory=directory,
        modelfolder=subfolder,
        verbose=verbose
    )

    info(Text("All steps completed successfully!", style="bold green"))
