"""
Constants module for the statapp application.

This module provides constants used throughout the application.
"""

# Dataset constants
DATASET_PREFIX = "Dataset475_CURVAS_ANNO"
PATIENT_PREFIX = "CURVAS"
FILE_ENDING = ".nii.gz"
CHANNEL_NAMES = {
    "0": "CT"
}
LABELS = {
    "background": 0,
    "pancreas": 1,
    "kidney": 2,
    "liver": 3
}

# Patient sets
TRAIN_PATIENTS = ["001", "002", "009", "011", "015", "017", "021", "023", "031", "034", "035", "037", "038", "039", "031", "042", "043", "045", "046", "048"]
VALIDATION_PATIENTS = ["049", "051", "058", "059", "061"]
TEST_PATIENTS = ["003", "005", "007", "008", "010", "013", "018", "020", "025", "026", "027", "028", "030", "032", "052", "053", "054", "055", "057", "062", "064", "066", "067", "069", "070", "071", "073", "075", "076", "077", "078", "080", "081", "082", "083", "084", "086", "087", "089", "090", "091", "092", "093", "094", "095", "096", "097", "098", "099", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115"]
