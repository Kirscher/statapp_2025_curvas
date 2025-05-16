"""
Metrics evaluation module for the statapp application.

This module provides a command to run metrics evaluation on model predictions,
computing various metrics such as DICE scores, calibration errors, and more.
"""

import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Union, Literal, Optional
import pandas as pd

import typer
from rich.text import Text

from statapp.utils import s3_utils
from statapp.utils.progress_tracker import track_progress
from statapp.utils.utils import info, setup_logging
from statapp.utils.upload_utils import upload_directory_to_s3
from statapp.core.constants import TRAIN_PATIENTS, VALIDATION_PATIENTS, TEST_PATIENTS
from statapp.core.metrics import apply_metrics, getting_gt

app = typer.Typer()

def get_selected_patients(patients: Union[List[str], Literal["all", "train", "validation", "test"]], verbose: bool = False) -> List[str]:
    """
    Get a list of selected patients based on the input.

    Args:
        patients: List of patient numbers or a predefined set ("all", "train", "validation", "test")
        verbose (bool): Enable verbose logging

    Returns:
        List[str]: List of selected patient numbers
    """
    # Set up logger
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

def get_available_models() -> List[str]:
    """
    Get a list of available models from S3.

    Returns:
        List[str]: List of model folder names
    """
    # List contents of the output directory
    contents = s3_utils.list_output_directory()

    # Extract folder names that match the pattern UKCHLL[NNN]/[MODEL_NAME]
    pattern = re.compile(r'^' + re.escape(os.environ['S3_OUTPUT_DIR']) + r'/UKCHLL\d{3}/([^/]+)(?:/.*)?$')
    available_models = set()

    for item in contents:
        key = item['Key']
        match = pattern.match(key)
        if match:
            model_name = match.group(1)
            # Exclude ensemble folders
            if not model_name.startswith("ensemble_"):
                available_models.add(model_name)

    return sorted(list(available_models))

def download_patient_data(patient_id: str, model_name: str, local_dir: Path, verbose: bool = False, progress_tracker = None) -> dict:
    """
    Download patient data and model predictions from S3.

    Args:
        patient_id (str): Patient ID (e.g., 001)
        model_name (str): Model name (e.g., anno1_init112233_foldall)
        local_dir (Path): Local directory to download the data to
        verbose (bool): Enable verbose logging
        progress_tracker: Optional progress tracker instance

    Returns:
        dict: Dictionary with paths to downloaded files
    """
    logger = setup_logging(verbose)

    # Create patient and model directories
    patient_dir = local_dir / f"UKCHLL{patient_id}"
    model_dir = patient_dir / model_name
    gt_dir = patient_dir / f"{patient_id}_GT"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Define the remote paths
    pred_remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{patient_id}/{model_name}"
    gt_remote_dir = f"{os.environ['S3_DATA_DIR']}/UKCHLL{patient_id}"

    # List contents of the directories
    output_contents = s3_utils.list_output_directory()
    data_contents = s3_utils.list_data_directory()

    # Find prediction and probability files
    pred_file = None
    prob_file = None
    pred_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/pred_.*\.nii\.gz$')
    prob_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/proba_.*\.nii\.gz$')

    for item in output_contents:
        key = item['Key']
        if pred_pattern.match(key):
            pred_file = key
        elif prob_pattern.match(key):
            prob_file = key

    if not pred_file or not prob_file:
        logger.error(f"Prediction or probability file not found for patient UKCHLL{patient_id} and model {model_name}")
        return None

    # Find ground truth files
    gt_files = []
    image_file = None
    gt_pattern = re.compile(r'^' + re.escape(gt_remote_dir) + r'/annotation.*\.nii\.gz$')
    image_pattern = re.compile(r'^' + re.escape(gt_remote_dir) + r'/image\.nii\.gz$')

    for item in data_contents:
        key = item['Key']
        if gt_pattern.match(key):
            gt_files.append(key)
        elif image_pattern.match(key):
            image_file = key

    if not gt_files or not image_file:
        logger.error(f"Ground truth or image file not found for patient UKCHLL{patient_id}")
        return None

    # Download files
    files_to_download = [
        (pred_file, model_dir / os.path.basename(pred_file)),
        (prob_file, model_dir / os.path.basename(prob_file)),
        (image_file, gt_dir / os.path.basename(image_file))
    ]

    for gt_file in gt_files:
        files_to_download.append((gt_file, gt_dir / os.path.basename(gt_file)))

    # Start time for tracking
    start_time = time.time()

    # Download each file
    for remote_path, local_path in files_to_download:
        # Get the file size
        bucket, key = s3_utils.parse_remote_path(remote_path)
        file_size = s3_utils.get_file_size(bucket, key)

        # Create a callback function for progress updates
        def progress_callback(bytes_transferred):
            if progress_tracker:
                progress_tracker.update_file_progress(
                    bytes_transferred, 
                    file_size, 
                    f"Downloading {os.path.basename(remote_path)} for patient UKCHLL{patient_id}", 
                    start_time
                )

        # Download the file with progress tracking
        success = s3_utils.download_file(
            remote_path=remote_path,
            local_path=str(local_path),
            callback=progress_callback
        )

        if not success:
            logger.error(f"Failed to download {os.path.basename(remote_path)} for patient UKCHLL{patient_id}")
            return None

    logger.info(f"All files downloaded successfully for patient UKCHLL{patient_id} and model {model_name}")

    # Return paths to downloaded files
    return {
        "pred": str(model_dir / os.path.basename(pred_file)),
        "prob": str(model_dir / os.path.basename(prob_file)),
        "name": model_name,
        "gt_dir": str(gt_dir)
    }

@app.command()
def compute_metrics(
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    models: List[str] = typer.Option(["all"], "-m", "--models", help="List of models to use for metrics computation (e.g., anno1_init112233_foldall) or 'all'"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """
    Compute metrics for model predictions on patient data.

    Downloads patient data and model predictions, computes various metrics,
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

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        download_dir = temp_path / "download"
        metrics_dir = temp_path / "metrics"

        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        # Create DataFrame to store metrics
        df = pd.DataFrame(columns=[
            "Patient_ID",
            "CT",
            "DICE_panc",
            "DICE_kidn",
            "DICE_livr",
            "CONF_panc",
            "CONF_kidn",
            "CONF_livr",
            "Entropy_GT",
            "Entropy_Pred",
            "Hausdorff_1",
            "Hausdorff_2",
            "Hausdorff_3",
            "ECE_1",
            "ECE_2",
            "ECE_3",
            "ACE_1",
            "ACE_2",
            "ACE_3",
            "AUROC_panc",
            "AUROC_kidn",
            "AUROC_livr",
            "AURC_panc",
            "AURC_kidn",
            "AURC_livr",
            "EAURC_panc",
            "EAURC_kidn",
            "EAURC_livr",
            "CRPS_panc",
            "CRPS_kidn",
            "CRPS_livr",
            "NCC_GT1-2",
            "NCC_GT1-3",
            "NCC_GT2-3",
            "NCC_mean"
        ])

        # Process each patient
        for patient_id in selected_patients:
            logger.info(f"Processing patient UKCHLL{patient_id}...")

            # Create patient metrics directory
            patient_metrics_dir = metrics_dir / f"UKCHLL{patient_id}"
            os.makedirs(patient_metrics_dir, exist_ok=True)

            # Create a list of patient-model pairs to process
            pairs = [(patient_id, model) for model in selected_models]

            # Define a function to get the combined size of all files for a patient-model pair
            def get_pair_size(pair):
                p_id, model_name = pair

                # Define the remote paths
                pred_remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{p_id}/{model_name}"
                gt_remote_dir = f"{os.environ['S3_DATA_DIR']}/UKCHLL{p_id}"

                # List contents of the directories
                output_contents = s3_utils.list_output_directory()
                data_contents = s3_utils.list_data_directory()

                # Count files and their sizes
                total_size = 0

                # Check prediction files
                pred_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/pred_.*\.nii\.gz$')
                prob_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/proba_.*\.nii\.gz$')

                for item in output_contents:
                    key = item['Key']
                    if pred_pattern.match(key) or prob_pattern.match(key):
                        bucket, file_key = s3_utils.parse_remote_path(key)
                        file_size = s3_utils.get_file_size(bucket, file_key)
                        total_size += file_size

                # Check ground truth files
                gt_pattern = re.compile(r'^' + re.escape(gt_remote_dir) + r'/annotation.*\.nii\.gz$')
                image_pattern = re.compile(r'^' + re.escape(gt_remote_dir) + r'/image\.nii\.gz$')

                for item in data_contents:
                    key = item['Key']
                    if gt_pattern.match(key) or image_pattern.match(key):
                        bucket, file_key = s3_utils.parse_remote_path(key)
                        file_size = s3_utils.get_file_size(bucket, file_key)
                        total_size += file_size

                return total_size

            # Define a function to process a patient-model pair
            def process_pair(pair, progress_tracker):
                p_id, model_name = pair
                # Get the actual size of the pair
                pair_size = get_pair_size(pair)
                progress_tracker.start_file(pair, f"Processing patient UKCHLL{p_id} with model {model_name}", pair_size)
                logger.info(f"Processing patient UKCHLL{p_id} with model {model_name}...")

                try:
                    # Download patient data and model predictions
                    model_files = download_patient_data(p_id, model_name, download_dir, verbose, progress_tracker)

                    if not model_files:
                        logger.error(f"Failed to download data for patient UKCHLL{p_id} and model {model_name}")
                        progress_tracker.complete_file(f"Failed to process patient UKCHLL{p_id} with model {model_name}", pair_size, time.time(), success=False)
                        return

                    # Load ground truth data
                    ct_image, annotations = getting_gt(model_files["gt_dir"])

                    # Compute metrics
                    logger.info(f"Computing metrics for patient UKCHLL{p_id} and model {model_name}...")
                    metrics = apply_metrics(model_files, ct_image, annotations)

                    # Add patient ID to metrics
                    metrics["Patient_ID"] = p_id

                    # Add metrics to DataFrame
                    nonlocal df
                    df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)

                    progress_tracker.complete_file(f"Processed patient UKCHLL{p_id} with model {model_name}", pair_size, time.time(), success=True)

                except Exception as e:
                    logger.error(f"Error processing patient UKCHLL{p_id} with model {model_name}: {str(e)}")
                    progress_tracker.complete_file(f"Error processing patient UKCHLL{p_id} with model {model_name}", pair_size, time.time(), success=False)

            # Calculate the total number of files to download
            def count_pair_files(pair):
                p_id, model_name = pair

                # Define the remote paths
                pred_remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{p_id}/{model_name}"
                gt_remote_dir = f"{os.environ['S3_DATA_DIR']}/UKCHLL{p_id}"

                # List contents of the directories
                output_contents = s3_utils.list_output_directory()
                data_contents = s3_utils.list_data_directory()

                # Count files
                file_count = 0

                # Check prediction files
                pred_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/pred_.*\.nii\.gz$')
                prob_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/proba_.*\.nii\.gz$')

                for item in output_contents:
                    key = item['Key']
                    if pred_pattern.match(key) or prob_pattern.match(key):
                        file_count += 1

                # Check ground truth files
                gt_pattern = re.compile(r'^' + re.escape(gt_remote_dir) + r'/annotation.*\.nii\.gz$')
                image_pattern = re.compile(r'^' + re.escape(gt_remote_dir) + r'/image\.nii\.gz$')

                for item in data_contents:
                    key = item['Key']
                    if gt_pattern.match(key) or image_pattern.match(key):
                        file_count += 1

                return file_count

            # Calculate the total number of files to download
            total_files = sum(count_pair_files(pair) for pair in pairs)

            logger.info(f"This will download a total of {total_files} files for patient UKCHLL{patient_id}")

            # Track progress of processing pairs
            track_progress(pairs, get_pair_size, process_pair, total_files)

            # Save metrics for this patient
            patient_metrics_file = patient_metrics_dir / "metrics.csv"
            patient_df = df[df["Patient_ID"] == patient_id]
            patient_df.to_csv(str(patient_metrics_file), index=False)

            # Upload metrics to S3
            logger.info(f"Uploading metrics for patient UKCHLL{patient_id} to S3...")

            def upload():
                upload_directory_to_s3(
                    directory=str(patient_metrics_dir),
                    remote_dir_env_var="S3_METRICS_DIR",
                    subfolder=f"UKCHLL{patient_id}",
                    verbose=verbose,
                    command_description=f"Upload metrics for patient UKCHLL{patient_id}",
                    tracker=False
                )
                logger.info(f"Metrics for patient UKCHLL{patient_id} uploaded successfully.")

            upload_thread = threading.Thread(target=upload, name="Uploader", args=())
            upload_thread.start()
            upload_thread.join()  # Wait for upload to complete

        # Save all metrics to a single file
        all_metrics_file = metrics_dir / "metrics.csv"
        df.to_csv(str(all_metrics_file), index=False)

        logger.info(Text("Metrics computation completed successfully", style="bold green"))
