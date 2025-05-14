"""
Predict outputs module for the statapp application.

This module provides a command to predict, for every given model, classes and softmaxs.
"""

import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Union, Literal, Optional

import typer
from rich.text import Text

from statapp.utils import s3_utils
from statapp.utils.empty_utils import empty_directory
from statapp.utils.progress_tracker import track_progress
from statapp.utils.upload_utils import upload_directory_to_s3
from statapp.utils.utils import info, setup_logging

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

# Constants from prepare.py
TRAIN_PATIENTS = ["001", "002", "009", "011", "015", "017", "021", "023", "031", "034", "035", "037", "038", "039", "031", "042", "043", "045", "046", "048"]
VALIDATION_PATIENTS = ["049", "051", "058", "059", "061"]
TEST_PATIENTS = ["003", "005", "007", "008", "010", "013", "018", "020", "025", "026", "027", "028", "030", "032", "052", "053", "054", "055", "057", "062", "064", "066", "067", "069", "070", "071", "073", "075", "076", "077", "078", "080", "081", "082", "083", "084", "086", "087", "089", "090", "091", "092", "093", "094", "095", "096", "097", "098", "099", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115"]

app = typer.Typer()

def get_dataset_code(patients: Union[List[str], Literal["all", "train", "validation", "test"]]) -> str:
    """
    Generate a dataset code based on the patient selection.

    Args:
        patients: List of patient numbers or a predefined set ("all", "train", "validation", "test")

    Returns:
        str: Dataset code for folder naming
    """
    if isinstance(patients, str):
        if patients in ["all", "train", "validation", "test"]:
            return patients

    # For custom patient lists, join the numbers
    return "".join(sorted(patients))

def get_available_models() -> List[str]:
    """
    Get a list of available models from S3.

    Returns:
        List[str]: List of model folder names
    """
    # List contents of the models directory
    contents = s3_utils.list_artifacts_directory()

    # Extract folder names that match the pattern anno[ANNOTATOR]_init[SEED]_foldall
    pattern = re.compile(r'^' + re.escape(os.environ['S3_ARTIFACTS_DIR']) + r'/' + 
                         re.escape(os.environ['S3_MODEL_ARTIFACTS_SUBDIR']) + 
                         r'/anno(\d+)_init(\d+)_foldall(?:/.*)?$')

    available_models = set()

    for item in contents:
        key = item['Key']
        match = pattern.match(key)
        if match:
            # Extract the model folder name (anno[ANNOTATOR]_init[SEED]_foldall)
            model_path = key.split('/')
            if len(model_path) >= 3:
                model_name = model_path[2]  # Get the third component (index 2)
                available_models.add(model_name)

    return sorted(list(available_models))

def download_model(model_name: str, local_models_dir: Path, verbose: bool = False, progress_tracker = None) -> Path:
    """
    Download a model checkpoint from S3.

    Args:
        model_name (str): Name of the model folder (e.g., anno1_init112233_foldall)
        local_models_dir (Path): Local directory to download the model to
        verbose (bool): Enable verbose logging
        progress_tracker: Optional progress tracker instance

    Returns:
        Path: Path to the downloaded model checkpoint
    """
    logger = setup_logging(verbose)

    # Create model directory
    model_dir = local_models_dir / model_name
    os.makedirs(model_dir, exist_ok=True)

    # Define the expected checkpoint path
    checkpoint_subdir = "nnUNetTrainer_Statapp__nnUNetPlans__3d_fullres"
    checkpoint_filename = "fold_all/checkpoint_final.pth"

    # Create the local directory structure
    local_checkpoint_dir = model_dir / checkpoint_subdir
    os.makedirs(local_checkpoint_dir, exist_ok=True)

    # Define the local and remote paths for the checkpoint file
    local_checkpoint_path = local_checkpoint_dir / checkpoint_filename
    remote_folder_path = f"{os.environ['S3_ARTIFACTS_DIR']}/{os.environ['S3_MODEL_ARTIFACTS_SUBDIR']}/{model_name}/{checkpoint_subdir}"
    remote_checkpoint_path = f"{remote_folder_path}/{checkpoint_filename}"

    # Download the model checkpoint
    logger.info(f"Downloading model checkpoint for {model_name}...")

    # Start time for tracking
    import time
    start_time = time.time()

    # Get the file size
    bucket, key = s3_utils.parse_remote_path(remote_checkpoint_path)
    file_size = s3_utils.get_file_size(bucket, key)

    # Create a callback function for progress updates
    def progress_callback(bytes_transferred):
        if progress_tracker:
            progress_tracker.update_file_progress(
                bytes_transferred, 
                file_size, 
                f"Downloading checkpoint for {model_name}", 
                start_time
            )

    # Download only the checkpoint file with progress tracking
    success = s3_utils.download_file(
        remote_path=remote_checkpoint_path,
        local_path=str(local_checkpoint_path),
        callback=progress_callback
    )

    # Check if the checkpoint file exists
    if not success or not local_checkpoint_path.exists():
        logger.error(f"Checkpoint file not found for model {model_name}")
        return None

    # Download additional small files to run the model
    for file in ["dataset.json", "plans.json"]:
        local_file = f"{local_checkpoint_dir}/{file}"

        success = s3_utils.download_file(
            remote_path=f"{remote_folder_path}/{file}",
            local_path=local_file,
            callback=progress_callback
        )

        if not success or not Path(local_file).exists():
            logger.error(f"Additional file {file} could not be downloaded.")
            return None




    logger.info(f"Model checkpoint downloaded successfully for {model_name}")
    return model_dir

def get_selected_patients(patients: Union[List[str], Literal["all", "train", "validation", "test"]], verbose: bool = False) -> List[str]:
    """
    Get a list of selected patients based on the input.

    Args:
        patients: List of patient numbers or a predefined set ("all", "train", "validation", "test")
        verbose (bool): Enable verbose logging

    Returns:
        List[str]: List of selected patient numbers
    """
    logger = setup_logging(verbose)

    # List contents of the data directory
    contents = s3_utils.list_data_directory()

    # Extract folder names that match the pattern UKCHLL[NNN]
    pattern = re.compile(r'^' + re.escape(os.environ['S3_DATA_DIR']) + r'/UKCHLL(\d{3})(?:/.*)?$')
    available_patients = {}

    for item in contents:
        key = item['Key']
        match = pattern.match(key)
        if match:
            patient_num = match.group(1)
            if patient_num not in available_patients:
                available_patients[patient_num] = []
            available_patients[patient_num].append(key)

    # Determine which patients to process based on the selection
    selected_patients = []

    if patients == "all":
        selected_patients = list(available_patients.keys())
        info(Text(f"Selected all {len(selected_patients)} patients", style="bold green"))
    elif patients == "train":
        for patient in TRAIN_PATIENTS:
            if patient in available_patients:
                selected_patients.append(patient)
        info(Text(f"Selected {len(selected_patients)} training patients", style="bold green"))
    elif patients == "validation":
        for patient in VALIDATION_PATIENTS:
            if patient in available_patients:
                selected_patients.append(patient)
        info(Text(f"Selected {len(selected_patients)} validation patients", style="bold green"))
    elif patients == "test":
        for patient in TEST_PATIENTS:
            if patient in available_patients:
                selected_patients.append(patient)
        info(Text(f"Selected {len(selected_patients)} test patients", style="bold green"))
    else:
        # Custom list of patients
        missing_patients = []
        for patient in patients:
            if patient in available_patients:
                selected_patients.append(patient)
            else:
                missing_patients.append(patient)

        if missing_patients:
            missing_list = ", ".join([f"UKCHLL{p}" for p in missing_patients])
            info(Text.assemble(("Warning: ", "bold yellow"), (f"The following patients were not found: {missing_list}", "")))

    if not selected_patients:
        info(Text.assemble(("Error: ", "bold red"), ("No valid patients selected", "")))
        return []

    return selected_patients

def download_patient_image(patient_id: str, local_dir: Path, verbose: bool = False, progress_tracker = None) -> Path:
    """
    Download a patient image from S3.

    Args:
        patient_id (str): Patient ID (e.g., 001)
        local_dir (Path): Local directory to download the image to
        verbose (bool): Enable verbose logging
        progress_tracker: Optional progress tracker instance

    Returns:
        Path: Path to the downloaded image
    """
    logger = setup_logging(verbose)

    # Create patient directory
    patient_dir = local_dir / f"UKCHLL{patient_id}"
    os.makedirs(patient_dir, exist_ok=True)

    # Define the remote path
    remote_dir = f"{os.environ['S3_DATA_DIR']}/UKCHLL{patient_id}"

    # List contents of the patient directory
    contents = s3_utils.list_data_directory()

    # Find the image file
    image_pattern = re.compile(r'^' + re.escape(remote_dir) + r'/image\.nii\.gz$')
    image_file = None

    for item in contents:
        key = item['Key']
        if image_pattern.match(key):
            image_file = key
            break

    if not image_file:
        logger.error(f"Image file not found for patient UKCHLL{patient_id}")
        return None

    # Download the image file
    local_image_path = patient_dir / f"CURVAS_{patient_id}_0000.nii.gz"
    logger.info(f"Downloading image for patient UKCHLL{patient_id}...")

    # Start time for tracking
    import time
    start_time = time.time()

    # Get the file size
    bucket, key = s3_utils.parse_remote_path(image_file)
    file_size = s3_utils.get_file_size(bucket, key)

    # Create a callback function for progress updates
    def progress_callback(bytes_transferred):
        if progress_tracker:
            progress_tracker.update_file_progress(
                bytes_transferred, 
                file_size, 
                f"Downloading image for patient UKCHLL{patient_id}", 
                start_time
            )

    # Download the file with progress tracking
    s3_utils.download_file(
        remote_path=image_file,
        local_path=str(local_image_path),
        callback=progress_callback
    )

    logger.info(f"Image downloaded successfully for patient UKCHLL{patient_id}")
    return local_image_path

@app.command()
def predict(
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    models: List[str] = typer.Option(["all"], "--models", help="List of models to use for prediction (e.g., anno1_init112233_foldall) or 'all'"),
    nb_workers: int = typer.Option(10, '-j', "--jobs", help="Number of processes to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
) -> None:
    """
    Predict segmentation for patients using specified models.

    Downloads patient images and model checkpoints, runs prediction for each model,
    and uploads results to S3.

    Patient selection options:
    - 'all': All available patients
    - 'train': Predefined training set
    - 'validation': Predefined validation set
    - 'test': Predefined test set
    - Custom list: Specific patient numbers (e.g., 001 034)

    Model selection options:
    - 'all': All available models
    - Custom list: Specific model names (e.g., anno1_init112233_foldall)
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Handle patient selection
    patient_selection = patients
    if len(patients) == 1:
        # Check if it's one of the predefined sets
        if patients[0] in ["all", "train", "validation", "test"]:
            patient_selection = patients[0]

    # Get selected patients
    selected_patients = get_selected_patients(patient_selection, verbose)
    if not selected_patients:
        return

    # Get available models
    available_models = get_available_models()
    if not available_models:
        logger.error("No models found in S3")
        return

    # Handle model selection
    selected_models = []
    if len(models) == 1 and models[0] == "all":
        selected_models = available_models
        info(Text(f"Selected all {len(selected_models)} models", style="bold green"))
    else:
        # Custom list of models
        missing_models = []
        for model in models:
            if model in available_models:
                selected_models.append(model)
            else:
                missing_models.append(model)

        if missing_models:
            missing_list = ", ".join(missing_models)
            info(Text.assemble(("Warning: ", "bold yellow"), (f"The following models were not found: {missing_list}", "")))

    if not selected_models:
        info(Text.assemble(("Error: ", "bold red"), ("No valid models selected", "")))
        return

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        models_dir = temp_path / "models"
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Download model checkpoints
        logger.info("Downloading model checkpoints...")
        model_paths = {}

        # Define a function to get the "size" of a model (we'll just use 1 for each model)
        def get_model_size(model_name):
            return 1

        # Define a function to download a model
        def process_model(model_name, progress_tracker):
            progress_tracker.start_file(model_name, f"Downloading model {model_name}", 1)
            model_path = download_model(model_name, models_dir, verbose, progress_tracker)
            if model_path:
                model_paths[model_name] = model_path
                progress_tracker.complete_file(f"Downloaded model {model_name}", 1, time.time(), success=True)
            else:
                progress_tracker.complete_file(f"Failed to download model {model_name}", 1, time.time(), success=False)

        # Track progress of downloading models
        track_progress(selected_models, get_model_size, process_model)

        if not model_paths:
            logger.error("Failed to download any model checkpoints")
            return

        # Process each patient

        # Define a function to get the "size" of a patient (we'll just use 1 for each patient)
        def get_patient_size(patient_id):
            return 1

        # Define a function to process a patient
        def process_patient(patient_id, progress_tracker):
            progress_tracker.start_file(patient_id, f"Processing patient UKCHLL{patient_id}", 1)
            logger.info(f"Processing patient UKCHLL{patient_id}...")

            try:
                # Download patient image
                patient_image_path = download_patient_image(patient_id, input_dir, verbose, progress_tracker)
                if not patient_image_path:
                    logger.error(f"Failed to download image for patient UKCHLL{patient_id}")
                    progress_tracker.complete_file(f"Failed to process patient UKCHLL{patient_id}", 1, time.time(), success=False)
                    return

                # Create patient output directory
                patient_output_dir = output_dir / f"UKCHLL{patient_id}"
                os.makedirs(patient_output_dir, exist_ok=True)

                # Process each model
                for model_name, model_path in model_paths.items():
                    logger.info(f"Running prediction with model {model_name} for patient UKCHLL{patient_id}...")

                    # Create model-specific output directory
                    model_output_dir = patient_output_dir / model_name
                    os.makedirs(model_output_dir, exist_ok=True)

                    # Initialize predictor
                    predictor = nnUNetPredictor()
                    checkpoint_path = model_path / "nnUNetTrainer_Statapp__nnUNetPlans__3d_fullres"
                    predictor.initialize_from_trained_model_folder(
                        str(checkpoint_path),
                        ("all"),
                        checkpoint_name="checkpoint_final.pth"
                    )

                    # Run prediction
                    predictor.predict_from_files(
                        str(patient_image_path.parent),
                        str(model_output_dir),
                        num_processes_preprocessing=nb_workers,
                        num_processes_segmentation_export=nb_workers,
                        save_probabilities=True
                    )

                    # Rename output files to include model name
                    for file in os.listdir(model_output_dir):
                        file_path = model_output_dir / file
                        if file.endswith(".nii.gz"):
                            if "softmax" in file:
                                # This is a probability file
                                new_name = f"proba_{model_name}.nii.gz"
                            else:
                                # This is a prediction file
                                new_name = f"pred_{model_name}.nii.gz"

                            # Rename the file
                            os.rename(file_path, model_output_dir / new_name)


                # Upload patient results to S3
                logger.info(f"Uploading results for patient UKCHLL{patient_id} to S3...")

                # Upload to S3_OUTPUT_DIR/UKCHLL{patient_id}
                upload_directory_to_s3(
                    directory=str(patient_output_dir / model_name),
                    remote_dir_env_var="S3_OUTPUT_DIR",
                    subfolder=f"UKCHLL{patient_id}/{model_name}",
                    verbose=verbose,
                    command_description=f"Upload prediction results for patient UKCHLL{patient_id}",
                    tracker=False
                )

                # Mark patient as completed
                progress_tracker.complete_file(f"Processed patient UKCHLL{patient_id}", 1, time.time(), success=True)

            except Exception as e:
                logger.error(f"Error processing patient UKCHLL{patient_id}: {str(e)}")
                progress_tracker.complete_file(f"Error processing patient UKCHLL{patient_id}", 1, time.time(), success=False)

        # Track progress of processing patients
        track_progress(selected_patients, get_patient_size, process_patient)

        logger.info(Text("Prediction completed successfully", style="bold green"))
