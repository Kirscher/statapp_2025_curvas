"""
Ensemble outputs module for the statapp application.

This module provides a command to ensemble the output for each patients on the 3 initilizations provided.
"""

import os
import typer
from rich.text import Text
import os
import typer
from statapp.utils import s3_utils
from statapp.utils.empty_utils import empty_directory
from statapp.utils.progress_tracker import track_progress
from statapp.utils.upload_utils import upload_directory_to_s3
from statapp.utils.utils import info, setup_logging

# Constants
S3_root = "output"
local_root = "outputs_nnUnet_preds"
prefix_patient = "UKCHLL"
annotations = ["anno1", "anno2", "anno3"]
initializations = ["init112233", "init445566", "init778899"]
prediction_filename = "pred_{variant}.nii.gz"
softmax_filename = "CURVAS_{patient_id}.npz"

# Constants from prepare.py
TRAIN_PATIENTS = ["001", "002", "009", "011", "015", "017", "021", "023", "031", "034", "035", "037", "038", "039", "031", "042", "043", "045", "046", "048"]
VALIDATION_PATIENTS = ["049", "051", "058", "059", "061"]
TEST_PATIENTS = ["003", "005", "007", "008", "010", "013", "018", "020", "025", "026", "027", "028", "030", "032", "052", "053", "054", "055", "057", "062", "064", "066", "067", "069", "070", "071", "073", "075", "076", "077", "078", "080", "081", "082", "083", "084", "086", "087", "089", "090", "091", "092", "093", "094", "095", "096", "097", "098", "099", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115"]

app = typer.Typer()

def download_pred_files(patient_id) :
    """
    Download softmaxs and predictions for a patient from S3
    NB : pourra se faire rapidos avec du threading après, mais déja bon bref on verra
    Args:

    patient_id : format : XXX

    Returns:
       
    """

    for anno in annotations:
        for init in initializations:
            variant = f"{anno}_{init}"
            s3_subdir = f"{S3_root}/{patient_id}/{variant}_foldall"
            local_subdir = os.path.join(local_root, patient_id, variant)

            os.makedirs(local_subdir, exist_ok=True)

            # Paths des fichiers à télécharger
            s3_pred = f"{s3_subdir}/{prediction_filename.format(variant=variant+'_foldall')}"
            s3_softmax = f"{s3_subdir}/{softmax_filename.format(patient_id=patient_id)}"

            local_pred = os.path.join(local_subdir, os.path.basename(s3_pred))
            local_softmax = os.path.join(local_subdir, os.path.basename(s3_softmax))

            # Téléchargement
            print(f"Downloading files for {patient_id} - {variant}")
            s3_utils.download_file(s3_pred, local_pred)
            s3_utils.download_file(s3_softmax, local_softmax)


def download_all_patients(patient_ids: list):
    """
    Boucle sur une liste de patients et télécharge tous les fichiers nécessaires.
    """
    for patient_id in patient_ids:
        download_pred_files(patient_id)
