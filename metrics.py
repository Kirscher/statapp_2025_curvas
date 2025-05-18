import pandas as pd
import numpy as np
import nibabel as nib
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, CropForeground
from torchmetrics.classification import MulticlassCalibrationError
from scipy.stats import norm
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils import column_or_1d
from sklearn.preprocessing import label_binarize

#faire un coup de subprocess pour ne pas avoir à pip install chacune des fonctions