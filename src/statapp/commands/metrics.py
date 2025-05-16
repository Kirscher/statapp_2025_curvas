"""
Metrics evaluation module for the statapp application.

This module provides a command to run metrics evaluation on model predictions,
computing various metrics such as DICE scores, calibration errors, and more.
"""

import os
import re
import tempfile
import time
import pandas as pd
from pathlib import Path
from typing import List, Union, Literal, Optional

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

def download_patient_data(patient_id: str, output_dir: Path, verbose: bool = False, progress_tracker = None) -> bool:
    """
    Download patient data (ground truth and predictions) from S3.

    Args:
        patient_id (str): Patient ID (e.g., 001)
        output_dir (Path): Output directory to save the files
        verbose (bool): Enable verbose logging
        progress_tracker: Optional progress tracker instance

    Returns:
        bool: True if files were downloaded successfully, False otherwise
    """
    logger = setup_logging(verbose)

    # Create patient directory
    patient_dir = output_dir / f"UKCHLL{patient_id}"
    os.makedirs(patient_dir, exist_ok=True)

    # Create ground truth directory
    gt_dir = patient_dir / f"UKCHLL{patient_id}_GT"
    os.makedirs(gt_dir, exist_ok=True)

    # Define the remote paths
    pred_remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{patient_id}"
    gt_remote_dir = f"{os.environ['S3_DATA_DIR']}/UKCHLL{patient_id}"

    # List contents of the output and data directories
    output_contents = s3_utils.list_output_directory()
    data_contents = s3_utils.list_data_directory()

    # Find all prediction files for this patient
    pred_files = []
    pred_pattern = re.compile(r'^' + re.escape(pred_remote_dir) + r'/')

    for item in output_contents:
        key = item['Key']
        if pred_pattern.match(key):
            pred_files.append(key)

    # Find all ground truth files for this patient
    gt_files = []
    gt_pattern = re.compile(r'^' + re.escape(gt_remote_dir) + r'/')

    for item in data_contents:
        key = item['Key']
        if gt_pattern.match(key):
            gt_files.append(key)

    if not pred_files:
        logger.error(f"No prediction files found for patient UKCHLL{patient_id}")
        return False

    if not gt_files:
        logger.error(f"No ground truth files found for patient UKCHLL{patient_id}")
        return False

    # Download the prediction files
    success = True
    start_time = time.time()

    # Download each prediction file
    for file_path in pred_files:
        # Get the file name from the path
        file_name = os.path.basename(file_path)

        # Get the model name from the path (if it exists)
        path_parts = file_path.split('/')
        if len(path_parts) > 3:  # Should be at least S3_OUTPUT_DIR/UKCHLL{patient_id}/model_name/file
            model_name = path_parts[3]
            # Create model directory if it doesn't exist
            model_dir = patient_dir / model_name
            os.makedirs(model_dir, exist_ok=True)
            local_file_path = model_dir / file_name
        else:
            local_file_path = patient_dir / file_name

        # Get the file size
        bucket, key = s3_utils.parse_remote_path(file_path)
        file_size = s3_utils.get_file_size(bucket, key)

        logger.info(f"Downloading {file_name} for patient UKCHLL{patient_id}")

        # Create a callback function for progress updates
        def file_progress_callback(bytes_transferred):
            if progress_tracker:
                progress_tracker.update_file_progress(
                    bytes_transferred, 
                    file_size, 
                    f"Downloading {file_name} for patient UKCHLL{patient_id}", 
                    start_time
                )

        # Download the file with progress tracking
        file_success = s3_utils.download_file(
            remote_path=file_path,
            local_path=str(local_file_path),
            callback=file_progress_callback
        )

        if not file_success:
            logger.error(f"Failed to download {file_name} for patient UKCHLL{patient_id}")
            success = False

    # Download each ground truth file
    for file_path in gt_files:
        # Get the file name from the path
        file_name = os.path.basename(file_path)
        local_file_path = gt_dir / file_name

        # Get the file size
        bucket, key = s3_utils.parse_remote_path(file_path)
        file_size = s3_utils.get_file_size(bucket, key)

        logger.info(f"Downloading {file_name} for patient UKCHLL{patient_id} ground truth")

        # Create a callback function for progress updates
        def file_progress_callback(bytes_transferred):
            if progress_tracker:
                progress_tracker.update_file_progress(
                    bytes_transferred, 
                    file_size, 
                    f"Downloading {file_name} for patient UKCHLL{patient_id} ground truth", 
                    start_time
                )

        # Download the file with progress tracking
        file_success = s3_utils.download_file(
            remote_path=file_path,
            local_path=str(local_file_path),
            callback=file_progress_callback
        )

        if not file_success:
            logger.error(f"Failed to download {file_name} for patient UKCHLL{patient_id} ground truth")
            success = False

    logger.info(f"All files downloaded successfully for patient UKCHLL{patient_id}")
    return success

@app.command()
def run_metrics(
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    models: List[str] = typer.Option(["all"], "-m", "--models", help="List of models to evaluate metrics for (e.g., anno1_init112233_foldall) or 'all'"),
    nb_workers: int = typer.Option(10, '-j', "--jobs", help="Number of processes to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
) -> None:
    """
    Run metrics evaluation on model predictions.

    Downloads patient ground truth and model predictions, computes metrics,
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

    # Process each patient
    for patient_id in selected_patients:
        logger.info(f"Processing patient UKCHLL{patient_id}...")

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            download_dir = temp_path / "download"
            metrics_dir = temp_path / "metrics"

            os.makedirs(download_dir, exist_ok=True)
            os.makedirs(metrics_dir, exist_ok=True)

            # Download patient data
            logger.info(f"Downloading data for patient UKCHLL{patient_id}...")

            # Define a function to get the "size" of a patient (we'll just use 1 for each patient)
            def get_patient_size(patient_id):
                return 1

            # Define a function to process a patient
            def process_patient(patient_id, progress_tracker):
                progress_tracker.start_file(patient_id, f"Processing patient UKCHLL{patient_id}", 1)
                logger.info(f"Processing patient UKCHLL{patient_id}...")

                try:
                    # Download patient data
                    success = download_patient_data(patient_id, download_dir, verbose, progress_tracker)
                    if not success:
                        logger.error(f"Failed to download data for patient UKCHLL{patient_id}")
                        progress_tracker.complete_file(f"Failed to process patient UKCHLL{patient_id}", 1, time.time(), success=False)
                        return

                    # Compute metrics
                    logger.info(f"Computing metrics for patient UKCHLL{patient_id}...")

                    # Prepare paths
                    patient_dir = download_dir / f"UKCHLL{patient_id}"
                    gt_dir = patient_dir / f"UKCHLL{patient_id}_GT"

                    # Get model directories
                    model_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and d.name != f"UKCHLL{patient_id}_GT"]
                    if not model_dirs:
                        logger.error(f"No model directories found for patient UKCHLL{patient_id}")
                        progress_tracker.complete_file(f"Failed to process patient UKCHLL{patient_id}", 1, time.time(), success=False)
                        return

                    # Prepare output dataframe
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

                    # Get ground truth
                    ct_img, annot = getting_gt(str(gt_dir))

                    # Process each model
                    for model_dir in model_dirs:
                        model_name = model_dir.name
                        logger.info(f"Computing metrics for model {model_name}...")

                        # Prepare model dict
                        model_dict = {"pred": None, "prob": None, "name": model_name}

                        # Find prediction and probability files
                        for item in os.listdir(model_dir):
                            item_path = os.path.join(model_dir, item)
                            if item.endswith(".nii.gz"):
                                model_dict["pred"] = item_path
                            elif item.endswith(".npz"):
                                model_dict["prob"] = item_path

                        # Compute metrics
                        if model_dict["pred"] is not None:
                            current_line = pd.DataFrame([apply_metrics(model_dict, ct_img, annot)])
                            df = pd.concat([df, current_line], ignore_index=True)
                        else:
                            logger.warning(f"No prediction file found for model {model_name}")

                    # Set patient ID
                    df['Patient_ID'] = f"UKCHLL{patient_id}"

                    # Save metrics to CSV
                    metrics_file = metrics_dir / f"metrics_UKCHLL{patient_id}.csv"
                    df.to_csv(metrics_file, index=False)
                    logger.info(f"Metrics saved to {metrics_file}")

                    # Upload metrics to S3
                    logger.info(f"Uploading metrics for patient UKCHLL{patient_id} to S3...")
                    upload_directory_to_s3(
                        directory=str(metrics_dir),
                        remote_dir_env_var="S3_OUTPUT_DIR",
                        subfolder=f"metrics/UKCHLL{patient_id}",
                        verbose=verbose,
                        command_description=f"Upload metrics for patient UKCHLL{patient_id}",
                        tracker=False
                    )

                    # Mark patient as completed
                    progress_tracker.complete_file(f"Processed patient UKCHLL{patient_id}", 1, time.time(), success=True)

                except Exception as e:
                    logger.error(f"Error processing patient UKCHLL{patient_id}: {str(e)}")
                    progress_tracker.complete_file(f"Error processing patient UKCHLL{patient_id}", 1, time.time(), success=False)

            # Track progress of processing patients
            track_progress([patient_id], get_patient_size, process_patient)

    logger.info(Text("Metrics evaluation completed successfully", style="bold green"))
