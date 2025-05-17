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
from typing import List, Union, Literal, Dict, Any, Optional

import pandas as pd
import typer
from rich.text import Text

from statapp.core.constants import TRAIN_PATIENTS, VALIDATION_PATIENTS, TEST_PATIENTS
from statapp.core.metrics import apply_metrics, getting_gt
from statapp.utils import s3_utils
from statapp.utils.progress_tracker import ProgressTracker
from statapp.utils.upload_utils import upload_directory_to_s3
from statapp.utils.utils import info, setup_logging

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


def find_ground_truth_files(patient_id: str, verbose: bool = False) -> Dict[str, str]:
    """
    Find ground truth files for a patient.

    Args:
        patient_id (str): Patient ID (e.g., 075)
        verbose (bool): Enable verbose logging

    Returns:
        Dict[str, str]: Dictionary with paths to ground truth files
    """
    logger = setup_logging(verbose)

    # Define the remote path
    gt_remote_dir = f"{os.environ['S3_DATA_DIR']}/UKCHLL{patient_id}"

    # List contents of the data directory
    data_contents = s3_utils.list_data_directory()

    # Find ground truth files using the standardized naming conventions
    gt_files = []
    image_file = None
    gt_pattern = re.compile(r'^' + re.escape(gt_remote_dir) + r'/annotation_[1-3]\.nii\.gz$')
    image_pattern = re.compile(r'^' + re.escape(gt_remote_dir) + r'/image\.nii\.gz$')

    for item in data_contents:
        key = item['Key']
        if gt_pattern.match(key):
            gt_files.append(key)
            logger.info(f"Found ground truth file: {key}")
        elif image_pattern.match(key):
            image_file = key
            logger.info(f"Found image file: {key}")

    # If we don't have exactly 3 ground truth files or an image file, we can't proceed
    if len(gt_files) != 3 or not image_file:
        logger.error(f"Ground truth files or image file not found for patient UKCHLL{patient_id}")
        logger.error(f"Found {len(gt_files)} ground truth files, expected 3")
        return {}

    return {
        "gt_files": gt_files,
        "image_file": image_file
    }


def download_ground_truth(patient_id: str, local_dir: Path, verbose: bool = False, 
                         progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, Any]:
    """
    Download ground truth files for a patient.

    Args:
        patient_id (str): Patient ID (e.g., 075)
        local_dir (Path): Local directory to download the data to
        verbose (bool): Enable verbose logging
        progress_tracker: Optional progress tracker instance

    Returns:
        Dict[str, Any]: Dictionary with paths to downloaded files and loaded data
    """
    logger = setup_logging(verbose)

    # Create patient and ground truth directories
    patient_dir = local_dir / f"UKCHLL{patient_id}"
    gt_dir = patient_dir / f"{patient_id}_GT"
    os.makedirs(gt_dir, exist_ok=True)

    # Find ground truth files
    gt_files_dict = find_ground_truth_files(patient_id, verbose)
    if not gt_files_dict:
        return {}

    # Prepare files to download with proper remote_path format
    bucket = os.environ['S3_BUCKET']
    files_to_download = []

    # Add image file
    image_key = gt_files_dict["image_file"]
    image_local_path = gt_dir / os.path.basename(image_key)
    files_to_download.append({
        'type': 'image',
        'patient': patient_id,
        'remote_path': f"{bucket}/{image_key}",
        'local_path': str(image_local_path),
        'display_name': f"image for UKCHLL{patient_id}"
    })

    # Add ground truth files
    for gt_file in gt_files_dict["gt_files"]:
        gt_local_path = gt_dir / os.path.basename(gt_file)
        files_to_download.append({
            'type': 'annotation',
            'patient': patient_id,
            'remote_path': f"{bucket}/{gt_file}",
            'local_path': str(gt_local_path),
            'display_name': f"annotation for UKCHLL{patient_id}"
        })

    # Function to get file size
    def get_file_size(file_info):
        bucket, key = s3_utils.parse_remote_path(file_info['remote_path'])
        return s3_utils.get_file_size(bucket, key)

    # Download each file
    for file_info in files_to_download:
        try:
            # Get the file size
            file_size = get_file_size(file_info)
            if file_size == 0:
                logger.warning(f"File size is 0 for {file_info['remote_path']}, file may not exist")
                # Instead of aborting, try to download the file anyway
                # The file might exist but head_object might be failing
                file_size = 1024  # Set a default size for progress tracking

            # Record start time
            start_time = time.time()

            # Initialize file progress tracking
            if progress_tracker:
                progress_tracker.start_file(
                    file_info['remote_path'],
                    f"Downloading {file_info['display_name']}",
                    file_size
                )

            # Create a callback function for progress updates
            def progress_callback(bytes_transferred):
                if progress_tracker:
                    progress_tracker.update_file_progress(
                        bytes_transferred,
                        file_size,
                        f"Downloading {file_info['display_name']}",
                        start_time
                    )

            # Download the file with progress tracking
            success = s3_utils.download_file(
                remote_path=file_info['remote_path'],
                local_path=file_info['local_path'],
                callback=progress_callback
            )

            # Mark file as complete in progress tracker
            if progress_tracker:
                if success:
                    progress_tracker.complete_file(
                        f"Downloaded {file_info['display_name']}",
                        file_size,
                        start_time,
                        success=True
                    )
                else:
                    progress_tracker.complete_file(
                        f"Failed to download {file_info['display_name']}",
                        file_size,
                        start_time,
                        success=False
                    )

            if not success:
                logger.error(f"Failed to download {file_info['display_name']}")
                return {}
        except Exception as e:
            logger.error(f"Error downloading file from {file_info['remote_path']} to {file_info['local_path']}: {str(e)}")
            # Mark file as failed in progress tracker
            if progress_tracker:
                progress_tracker.complete_file(
                    f"Error downloading {file_info['display_name']}",
                    file_size,
                    start_time,
                    success=False
                )
            return {}

    logger.info(f"All ground truth files downloaded successfully for patient UKCHLL{patient_id}")

    # Load ground truth data
    try:
        ct_image, annotations = getting_gt(str(gt_dir))
        logger.info(f"Ground truth data loaded successfully for patient UKCHLL{patient_id}")
    except Exception as e:
        logger.error(f"Error loading ground truth data for patient UKCHLL{patient_id}: {str(e)}")
        return {}

    return {
        "gt_dir": str(gt_dir),
        "ct_image": ct_image,
        "annotations": annotations
    }


def find_model_files(patient_id: str, model_name: str, verbose: bool = False) -> Dict[str, str]:
    """
    Find model prediction and probability files for a patient and model.

    Args:
        patient_id (str): Patient ID (e.g., 075)
        model_name (str): Model name (e.g., anno1_init112233_foldall)
        verbose (bool): Enable verbose logging

    Returns:
        Dict[str, str]: Dictionary with paths to model files
    """
    logger = setup_logging(verbose)

    # Define the remote path
    pred_remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{patient_id}/{model_name}"

    # List contents of the output directory
    output_contents = s3_utils.list_output_directory()

    # Find prediction and probability files based on the standardized naming conventions
    pred_file = None
    prob_file = None

    # Define patterns for the standardized file naming conventions
    pred_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/pred_.*\.nii\.gz$')
    curvas_nii_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/CURVAS_' + re.escape(patient_id) + r'\.nii\.gz$')
    curvas_npz_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/CURVAS_' + re.escape(patient_id) + r'\.npz$')

    # Collect all .nii.gz files as fallback
    nii_files = []

    for item in output_contents:
        key = item['Key']
        if pred_pattern.match(key):
            pred_file = key
            logger.info(f"Found prediction file: {key}")
        elif curvas_nii_pattern.match(key):
            # CURVAS .nii.gz file can be used as prediction file
            pred_file = key
            logger.info(f"Found CURVAS .nii.gz file: {key}")
        elif curvas_npz_pattern.match(key):
            # CURVAS .npz file is used as probability file
            prob_file = key
            logger.info(f"Found CURVAS .npz file: {key}")
        elif key.endswith('.nii.gz') and key.startswith(pred_remote_dir):
            nii_files.append(key)
            logger.info(f"Found .nii.gz file: {key}")

    # If we didn't find a specific prediction file, try to use any .nii.gz file
    if not pred_file and nii_files:
        pred_file = nii_files[0]
        logger.info(f"Using .nii.gz file as prediction: {pred_file}")

    # If we still don't have a prediction file, we can't proceed
    if not pred_file:
        logger.error(f"Prediction file not found for patient UKCHLL{patient_id} and model {model_name}")
        return {}

    # If no CURVAS .npz file is found, look for any .npz file
    if not prob_file:
        npz_files = [item['Key'] for item in output_contents if item['Key'].endswith('.npz') and item['Key'].startswith(pred_remote_dir)]
        if npz_files:
            prob_file = npz_files[0]
            logger.info(f"Using .npz file as probability file: {prob_file}")
        else:
            # If no .npz file is found, we can't proceed
            logger.error(f"Probability file not found for patient UKCHLL{patient_id} and model {model_name}")
            return {}

    return {
        "pred_file": pred_file,
        "prob_file": prob_file
    }


def download_model_files(patient_id: str, model_name: str, local_dir: Path, verbose: bool = False,
                        progress_tracker: Optional[ProgressTracker] = None) -> Dict[str, str]:
    """
    Download model prediction and probability files for a patient and model.

    Args:
        patient_id (str): Patient ID (e.g., 075)
        model_name (str): Model name (e.g., anno1_init112233_foldall)
        local_dir (Path): Local directory to download the data to
        verbose (bool): Enable verbose logging
        progress_tracker: Optional progress tracker instance

    Returns:
        Dict[str, str]: Dictionary with paths to downloaded files
    """
    logger = setup_logging(verbose)

    # Create patient and model directories
    patient_dir = local_dir / f"UKCHLL{patient_id}"
    model_dir = patient_dir / model_name
    os.makedirs(model_dir, exist_ok=True)

    # Find model files
    model_files_dict = find_model_files(patient_id, model_name, verbose)
    if not model_files_dict:
        return {}

    # Prepare files to download with proper remote_path format
    bucket = os.environ['S3_BUCKET']
    files_to_download = []

    # Add prediction file
    pred_key = model_files_dict["pred_file"]
    pred_local_path = model_dir / os.path.basename(pred_key)
    files_to_download.append({
        'type': 'prediction',
        'patient': patient_id,
        'model': model_name,
        'remote_path': f"{bucket}/{pred_key}",
        'local_path': str(pred_local_path),
        'display_name': f"prediction for UKCHLL{patient_id} and model {model_name}"
    })

    # Add probability file
    prob_key = model_files_dict["prob_file"]
    prob_local_path = model_dir / os.path.basename(prob_key)
    files_to_download.append({
        'type': 'probability',
        'patient': patient_id,
        'model': model_name,
        'remote_path': f"{bucket}/{prob_key}",
        'local_path': str(prob_local_path),
        'display_name': f"probability for UKCHLL{patient_id} and model {model_name}"
    })

    # Function to get file size
    def get_file_size(file_info):
        bucket, key = s3_utils.parse_remote_path(file_info['remote_path'])
        return s3_utils.get_file_size(bucket, key)

    # Download each file
    for file_info in files_to_download:
        try:
            # Get the file size
            file_size = get_file_size(file_info)
            if file_size == 0:
                logger.warning(f"File size is 0 for {file_info['remote_path']}, file may not exist")
                # Instead of aborting, try to download the file anyway
                # The file might exist but head_object might be failing
                file_size = 1024  # Set a default size for progress tracking

            # Record start time
            start_time = time.time()

            # Initialize file progress tracking
            if progress_tracker:
                progress_tracker.start_file(
                    file_info['remote_path'],
                    f"Downloading {file_info['display_name']}",
                    file_size
                )

            # Create a callback function for progress updates
            def progress_callback(bytes_transferred):
                if progress_tracker:
                    progress_tracker.update_file_progress(
                        bytes_transferred,
                        file_size,
                        f"Downloading {file_info['display_name']}",
                        start_time
                    )

            # Download the file with progress tracking
            success = s3_utils.download_file(
                remote_path=file_info['remote_path'],
                local_path=file_info['local_path'],
                callback=progress_callback
            )

            # Mark file as complete in progress tracker
            if progress_tracker:
                if success:
                    progress_tracker.complete_file(
                        f"Downloaded {file_info['display_name']}",
                        file_size,
                        start_time,
                        success=True
                    )
                else:
                    progress_tracker.complete_file(
                        f"Failed to download {file_info['display_name']}",
                        file_size,
                        start_time,
                        success=False
                    )

            if not success:
                logger.error(f"Failed to download {file_info['display_name']}")
                return {}
        except Exception as e:
            logger.error(f"Error downloading file from {file_info['remote_path']} to {file_info['local_path']}: {str(e)}")
            # Mark file as failed in progress tracker
            if progress_tracker:
                progress_tracker.complete_file(
                    f"Error downloading {file_info['display_name']}",
                    file_size,
                    start_time,
                    success=False
                )
            return {}

    logger.info(f"All model files downloaded successfully for patient UKCHLL{patient_id} and model {model_name}")

    return {
        "pred": str(model_dir / os.path.basename(model_files_dict["pred_file"])),
        "prob": str(model_dir / os.path.basename(model_files_dict["prob_file"])),
        "name": model_name
    }


@app.command()
def compute_metrics(
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 075 034) or 'all', 'train', 'validation', 'test'"),
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
    - Custom list: Specific patient numbers (e.g., 075 034)

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

            # Calculate total number of files to download
            # Ground truth files: 1 image file + 3 annotation files = 4 files
            # Model files: 1 prediction file + 1 probability file = 2 files per model
            total_files = 4 + (len(selected_models) * 2)

            # Create progress tracker with the correct total number of files
            progress_tracker = ProgressTracker(files=[], get_file_size=lambda _: 0, total_files=total_files)
            progress_tracker.start()  # Initialize progress bars

            # Download ground truth files once for this patient
            logger.info(f"Downloading ground truth files for patient UKCHLL{patient_id}...")
            gt_data = download_ground_truth(patient_id, download_dir, verbose, progress_tracker)

            if not gt_data:
                logger.error(f"Failed to download ground truth data for patient UKCHLL{patient_id}")
                continue

            # Process each model for this patient
            for model_name in selected_models:
                logger.info(f"Processing model {model_name} for patient UKCHLL{patient_id}...")

                # Download model files
                model_files = download_model_files(patient_id, model_name, download_dir, verbose, progress_tracker)

                if not model_files:
                    logger.error(f"Failed to download model files for patient UKCHLL{patient_id} and model {model_name}")
                    continue

                # Add ground truth directory to model files
                model_files["gt_dir"] = gt_data["gt_dir"]

                try:
                    # Compute metrics
                    logger.info(f"Computing metrics for patient UKCHLL{patient_id} and model {model_name}...")
                    metrics = apply_metrics(model_files, gt_data["ct_image"], gt_data["annotations"])

                    # Add patient ID to metrics
                    metrics["Patient_ID"] = patient_id

                    # Add metrics to DataFrame
                    df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)

                    logger.info(f"Metrics computed successfully for patient UKCHLL{patient_id} and model {model_name}")
                except Exception as e:
                    logger.error(f"Error computing metrics for patient UKCHLL{patient_id} and model {model_name}: {str(e)}")

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

            # Stop the progress tracker
            progress_tracker.stop()

        # Save all metrics to a single file
        all_metrics_file = metrics_dir / "metrics.csv"
        df.to_csv(str(all_metrics_file), index=False)

        logger.info(Text("Metrics computation completed successfully", style="bold green"))
