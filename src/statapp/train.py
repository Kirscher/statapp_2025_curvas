"""
Dataset training for the statapp application.

This module provides commands for training the model.
"""

from pathlib import Path
from rich.text import Text
import typer
from typing import Optional
from statapp.utils import setup_nnunet_env
import subprocess
import os

import statapp.utils as utils

app = typer.Typer()

@app.command()
def train(
    base_directory: str = typer.Argument(..., help="Local directory path to the dataset"),
    preprocessed_directory: str = typer.Option("nnUNet_preprocessed", "--preprocessed", "-p", help="Local path of preprocessed data relative to base"),
    result_directory: str = typer.Option("nnUNet_results", "--results", "-r", help="Local path of results relative to base"),
    dataset: str = typer.Option("", "--dataset", "-d", help="Dataset on wich to train")
) -> None:
    """
    Train a model on a given dataset
    """
    # We don't care about output directory as we are not using it here 
    setup_nnunet_env(Path(base_directory), Path(f""), Path(preprocessed_directory), Path(result_directory))

    if dataset == "":
        datasets = os.listdir(Path(base_directory)/Path(preprocessed_directory))
    else:
        datasets = [dataset]
    
    for dataset in datasets:
        # ignore hidden files
        if dataset.startswith('.'):
            continue


        command = [
            "nnUNetv2_train",
            f"{dataset}",  # Dataset ID
            "3d_fullres",  # Plan
            "all",  # Fold
            "--npz",
            "--c"
        ]
        subprocess.run(command)
