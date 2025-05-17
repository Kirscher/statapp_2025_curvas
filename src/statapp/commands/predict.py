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
import threading

import typer
from rich.text import Text

from statapp.utils import s3_utils
from statapp.utils.empty_utils import empty_directory
from statapp.utils.progress_tracker import track_progress
from statapp.utils.upload_utils import upload_directory_to_s3
from statapp.utils.utils import info, setup_logging
from statapp.core.constants import TRAIN_PATIENTS, VALIDATION_PATIENTS, TEST_PATIENTS

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

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
        progress_tracker: Optional progress tracker instance to use instead of creating a new one

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
    bucket = os.environ['S3_BUCKET']

    # Prepare file list for downloading
    files_to_download = [
        {
            'remote_path': f"{bucket}/{remote_folder_path}/{checkpoint_filename}",
            'local_path': str(local_checkpoint_path),
            'display_name': f"checkpoint for {model_name}"
        }
    ]

    # Add additional small files to run the model
    for file in ["dataset.json", "plans.json"]:
        local_file_path = local_checkpoint_dir / file
        files_to_download.append({
            'remote_path': f"{bucket}/{remote_folder_path}/{file}",
            'local_path': str(local_file_path),
            'display_name': f"{file} for {model_name}"
        })

    # Function to get file size
    def get_file_size(file_info):
        bucket, key = s3_utils.parse_remote_path(file_info['remote_path'])
        return s3_utils.get_file_size(bucket, key) or 1024  # Default to 1KB if size is 0

    # Function to process each file
    def process_file(file_info, progress_tracker):
        display_name = file_info['display_name']

        # Get the file size
        file_size = get_file_size(file_info)

        # Record start time
        start_time = time.time()

        try:
            # Start tracking file download
            progress_tracker.start_file(
                file_info['remote_path'],
                f"Downloading {display_name}",
                file_size
            )

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_info['local_path']), exist_ok=True)

            # Download file
            success = s3_utils.download_file(
                remote_path=file_info['remote_path'],
                local_path=file_info['local_path'],
                callback=progress_tracker.get_progress_callback(
                    f"Downloading {display_name}",
                    file_size,
                    start_time
                )
            )

            # Complete file download
            progress_tracker.complete_file(
                f"Downloaded {display_name}",
                file_size,
                start_time,
                success=success
            )

            if not success:
                logger.error(f"Failed to download {display_name}")
                return False

            return True
        except Exception as e:
            # Mark file as failed
            progress_tracker.complete_file(
                f"Error downloading {display_name}",
                file_size,
                start_time,
                success=False
            )
            logger.error(f"Error processing {display_name}: {str(e)}")
            return False

    # Track progress and process files
    try:
        # If a progress tracker is provided, use it directly
        if progress_tracker:
            success = True
            for file_info in files_to_download:
                if not process_file(file_info, progress_tracker):
                    success = False
        else:
            # Otherwise create a new one with track_progress
            success = track_progress(files_to_download, get_file_size, process_file)

        if not success:
            logger.error(f"Failed to download model files for {model_name}")
            return None

        # Check if the checkpoint file exists
        if not local_checkpoint_path.exists():
            logger.error(f"Checkpoint file not found for model {model_name}")
            return None

        logger.info(f"Model checkpoint downloaded successfully for {model_name}")
        return model_dir
    except Exception as e:
        logger.error(f"Error downloading model files: {str(e)}")
        return None

def get_selected_patients(patients: Union[List[str], Literal["all", "train", "validation", "test"]], verbose: bool = False) -> List[str]:
    """
    Get a list of selected patients based on the input.

    Args:
        patients: List of patient numbers or a predefined set ("all", "train", "validation", "test")
        verbose (bool): Enable verbose logging

    Returns:
        List[str]: List of selected patient numbers
    """

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
        progress_tracker: Optional progress tracker instance to use instead of creating a new one

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
    bucket = os.environ['S3_BUCKET']

    # Prepare file list for downloading
    files_to_download = [
        {
            'remote_path': f"{bucket}/{image_file}",
            'local_path': str(local_image_path),
            'display_name': f"image for patient UKCHLL{patient_id}"
        }
    ]

    # Function to get file size
    def get_file_size(file_info):
        bucket, key = s3_utils.parse_remote_path(file_info['remote_path'])
        return s3_utils.get_file_size(bucket, key) or 1024  # Default to 1KB if size is 0

    # Function to process each file
    def process_file(file_info, progress_tracker):
        display_name = file_info['display_name']

        # Get the file size
        file_size = get_file_size(file_info)

        # Record start time
        start_time = time.time()

        try:
            # Start tracking file download
            progress_tracker.start_file(
                file_info,
                f"Downloading {display_name}",
                file_size
            )

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_info['local_path']), exist_ok=True)

            # Download file
            success = s3_utils.download_file(
                remote_path=file_info['remote_path'],
                local_path=file_info['local_path'],
                callback=progress_tracker.get_progress_callback(
                    f"Downloading {display_name}",
                    file_size,
                    start_time
                )
            )

            # Complete file download
            progress_tracker.complete_file(
                f"Downloaded {display_name}",
                file_size,
                start_time,
                success=success
            )

            if not success:
                logger.error(f"Failed to download {display_name}")
                return False

            return True
        except Exception as e:
            # Mark file as failed
            progress_tracker.complete_file(
                f"Error downloading {display_name}",
                file_size,
                start_time,
                success=False
            )
            logger.error(f"Error processing {display_name}: {str(e)}")
            return False

    # Track progress and process files
    try:
        # If a progress tracker is provided, use it directly
        if progress_tracker:
            success = True
            for file_info in files_to_download:
                if not process_file(file_info, progress_tracker):
                    success = False
        else:
            # Otherwise create a new one with track_progress
            success = track_progress(files_to_download, get_file_size, process_file)

        if not success:
            logger.error(f"Failed to download image for patient UKCHLL{patient_id}")
            return None

        # Check if the image file exists
        if not local_image_path.exists():
            logger.error(f"Image file not found for patient UKCHLL{patient_id}")
            return None

        logger.info(f"Image downloaded successfully for patient UKCHLL{patient_id}")
        return local_image_path
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return None

@app.command()
def predict(
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    models: List[str] = typer.Option(["all"], "-m", "--models", help="List of models to use for prediction (e.g., anno1_init112233_foldall) or 'all'"),
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

        # Define a function to get the actual size of a model from S3
        def get_model_size(model_name):
            # Define the expected checkpoint path
            checkpoint_subdir = "nnUNetTrainer_Statapp__nnUNetPlans__3d_fullres"
            checkpoint_filename = "fold_all/checkpoint_final.pth"

            # Define the remote paths for the model files
            remote_folder_path = f"{os.environ['S3_ARTIFACTS_DIR']}/{os.environ['S3_MODEL_ARTIFACTS_SUBDIR']}/{model_name}/{checkpoint_subdir}"
            bucket = os.environ['S3_BUCKET']

            # List of files to check
            files_to_check = [
                f"{remote_folder_path}/{checkpoint_filename}",
                f"{remote_folder_path}/dataset.json",
                f"{remote_folder_path}/plans.json"
            ]

            # Calculate total size
            total_size = 0
            for file_path in files_to_check:
                remote_path = f"{bucket}/{file_path}"
                bucket_name, key = s3_utils.parse_remote_path(remote_path)
                size = s3_utils.get_file_size(bucket_name, key) or 0
                total_size += size

            # Return at least 1KB if no files were found
            return total_size or 1024

        # Define a function to download a model
        def process_model(model_name, progress_tracker):
            model_size = get_model_size(model_name)
            progress_tracker.start_file(model_name, f"Downloading model {model_name}", model_size)
            model_path = download_model(model_name, models_dir, verbose, progress_tracker)
            if model_path:
                model_paths[model_name] = model_path
                progress_tracker.complete_file(f"Downloaded model {model_name}", model_size, time.time(), success=True)
            else:
                progress_tracker.complete_file(f"Failed to download model {model_name}", model_size, time.time(), success=False)

        # Track progress of downloading models
        track_progress(selected_models, get_model_size, process_model)

        if not model_paths:
            logger.error("Failed to download any model checkpoints")
            return

        # Process each patient

        # Define a function to get the actual size of a patient image from S3
        def get_patient_size(patient_id):
            # Define the remote path
            remote_dir = f"{os.environ['S3_DATA_DIR']}/UKCHLL{patient_id}"

            # List contents of the data directory
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
                # Return a default size if the image file is not found
                return 50 * 1024 * 1024

            # Get the file size
            bucket = os.environ['S3_BUCKET']
            size = s3_utils.get_file_size(bucket, image_file) or 0

            # Return at least 1KB if the file size is 0
            return size or 1024

        # Define a function to process a patient
        def process_patient(patient_id, progress_tracker):
            patient_size = get_patient_size(patient_id)
            progress_tracker.start_file(patient_id, f"Processing patient UKCHLL{patient_id}", patient_size)
            logger.info(f"Processing patient UKCHLL{patient_id}...")

            try:
                # Download patient image
                patient_image_path = download_patient_image(patient_id, input_dir, verbose, progress_tracker)
                if not patient_image_path:
                    logger.error(f"Failed to download image for patient UKCHLL{patient_id}")
                    progress_tracker.complete_file(f"Failed to process patient UKCHLL{patient_id}", patient_size, time.time(), success=False)
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



                # Upload to S3_OUTPUT_DIR/UKCHLL{patient_id}
                def upload():
                    logger.info(f"Uploading results for patient UKCHLL{patient_id} to S3 in the background.")

                    upload_directory_to_s3(
                        directory=str(patient_output_dir / model_name),
                        remote_dir_env_var="S3_OUTPUT_DIR",
                        subfolder=f"UKCHLL{patient_id}/{model_name}",
                        verbose=verbose,
                        command_description=f"Upload prediction results for patient UKCHLL{patient_id}",
                        tracker=False
                    )

                    logger.info(f"Uploading results for patient UKCHLL{patient_id} to S3 done!")


                upload_thread = threading.Thread(target=upload, name="Downloader", args=())
                upload_thread.start()

                # Mark patient as completed
                progress_tracker.complete_file(f"Processed patient UKCHLL{patient_id}", patient_size, time.time(), success=True)

            except Exception as e:
                logger.error(f"Error processing patient UKCHLL{patient_id}: {str(e)}")
                progress_tracker.complete_file(f"Error processing patient UKCHLL{patient_id}", patient_size, time.time(), success=False)

        # Track progress of processing patients
        track_progress(selected_patients, get_patient_size, process_patient)

        logger.info(Text("Prediction completed successfully", style="bold green"))
