"""
Empty artifacts module for the statapp application.

This module provides a command to predict, for every given model, classes and softmaxs.
"""

import typer

from statapp.utils import s3_utils
from statapp.utils.empty_utils import empty_directory

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

app = typer.Typer()

@app.command()
def predict(
    input_folder: str = typer.Argument(..., help="test dataset"),
    output_folder: str = typer.Argument(..., help="prediction"),
    trained_model: str = typer.Argument(..., help="path to access the model"),
    nb_workers: int = typer.Option(10, '-j', "--jobs", help="number of processes to run")
) -> None:


    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(trained_model, None, checkpoint_name="checkpoint_best.pth")

    # Run prediction
    predictor.predict_from_files(
        input_folder,
        output_folder,
        num_processes_preprocessing=nb_workers,
        num_processes_segmentation_export=nb_workers,
        save_probabilities=True
    )


