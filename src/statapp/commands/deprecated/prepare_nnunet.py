"""
Dataset preparation module for the statapp application.

This module provides commands for preparing datasets for analysis.
"""

import subprocess
from pathlib import Path

import typer

from statapp.utils.utils import setup_nnunet_env

app = typer.Typer()

@app.command()
def deprecated_prepare_nnunet(
    base_directory: str = typer.Argument(..., help="Local directory path to the dataset"),
    dataset: int = typer.Argument(..., help="ID of the dataset to preprocess"),
    raw_directory: str = typer.Option("nnUNet_raw", "--raw", "-r", help="Local path of raw data relative to base"),
    preprocessed_directory: str = typer.Option("nnUNet_preprocessed", "--preprocessed", "-p", help="Local path of preprocessed data relative to base"),
) -> None:
    """
    [DEPRECATED] Prepare a dataset for analysis in the S3 data folder.
    """
    # We don't care about output directory as we are not using it here 
    setup_nnunet_env(Path(base_directory), Path(raw_directory), Path(preprocessed_directory), Path(f""))

    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d", f"{dataset}",
        "-c", "3d_fullres"
        "--verify_dataset_integrity",
        "-np",
        "2",
        "-npfp",
        "2"
    ]
    subprocess.run(command)