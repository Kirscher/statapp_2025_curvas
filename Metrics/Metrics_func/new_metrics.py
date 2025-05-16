"""
This code aims at computing metrics for nnUNet segmentation predictions. It has 3 parts : 
-Imports
-Functions (metric computation and preprocess functions)
-Body (applying the functions)

It requires a folder with for each patient a folder containing GT and predictions (see BODY part for the detail)
"""

#IMPORTS
"""
The needed imports for the metrics computation.
Please run : 
pip install scikit-learn torch scipy monai torchmetrics numba
to install the libraries that are not automatically implemented by onyxia.
"""

import SimpleITK as sitk
import os
import re
import numpy as np
import pandas as pd
from sklearn.utils import column_or_1d
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import torch
from scipy.stats import norm
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.transforms import CropForeground
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import entropy
from scipy.spatial.distance import directed_hausdorff
from torchmetrics.classification import MulticlassCalibrationError
from numba import njit

#FUNCTIONS

"""
Metrics computation functions :
- consensus_dice_score (DICE)
- compute_entropy (Entropies)
- compute_hausdorff_distances (Hausdorff distances)
- compute_auroc (AUROC)
- rc_curve_stats, calc_aurc, calc_eaurc, compute_aurc_eaurc (AURC and EAURC)
- expected_calibration_error, multirater_expected_calibration_error (ECE)
- calc_ace, calib_stats, prepare_inputs_for_ace (ACE)
- volume_metric, compute_probabilistic_volume, calculate_volumes_distributions, heaviside, crps_computation (CRPS)
- compute_ncc (NCC)

"""

'''
Dice Score Evaluation
'''

@njit
def compute_consensus_masks(groundtruth, num_organs=3):
    """
    Computes consensus, consensus background, and dissensus masks for all organs.
    
    Returns:
        consensus: shape (3, slices, X, Y)
        consensus_bck: same shape
        dissensus: same shape
    """
    consensus = np.zeros((num_organs, groundtruth.shape[1], groundtruth.shape[2], groundtruth.shape[3]), dtype=np.uint8)
    consensus_bck = np.zeros_like(consensus)
    dissensus = np.zeros_like(consensus)

    for i in range(num_organs):
        organ_val = i + 1
        for s in range(groundtruth.shape[1]):
            for x in range(groundtruth.shape[2]):
                for y in range(groundtruth.shape[3]):
                    # Check consensus on foreground
                    fg_agree = True
                    bck_agree = True
                    for annot in range(groundtruth.shape[0]):
                        if groundtruth[annot, s, x, y] != organ_val:
                            fg_agree = False
                        if groundtruth[annot, s, x, y] == organ_val:
                            bck_agree = False
                    consensus[i, s, x, y] = 1 if fg_agree else 0
                    consensus_bck[i, s, x, y] = 1 if bck_agree else 0
                    if not fg_agree and not bck_agree:
                        dissensus[i, s, x, y] = 1
    return consensus, consensus_bck, dissensus

def compute_consensus_dice_score(groundtruth, bin_pred, prob_pred):
    """
    Computes average dice score on consensus regions only.
    """

    # One-hot encoding of binarized predictions (shape: (4, slices, X, Y))
    prediction_onehot = AsDiscrete(to_onehot=4)(torch.from_numpy(np.expand_dims(bin_pred, axis=0)))[1:].numpy().astype(np.uint8)

    # Organ mapping
    organs = {1: 'panc', 2: 'kidn', 3: 'livr'}
    num_organs = 3

    # Compute consensus and dissensus masks
    consensus, consensus_bck, dissensus = compute_consensus_masks(groundtruth)

    predictions = {}
    groundtruth_consensus = {}
    confidence = {}
    dice_scores = {}

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=True)

    for i, organ_name in organs.items():
        idx = i - 1

        # Mask prediction and groundtruth using dissensus
        pred_masked = prediction_onehot[idx] * (1 - dissensus[idx])
        gt_masked = consensus[idx] * (1 - dissensus[idx])

        predictions[organ_name] = pred_masked
        groundtruth_consensus[organ_name] = gt_masked

        # Confidence: mean probability in consensus organ and background
        prob_organ = np.where(consensus[idx] == 1, prob_pred[idx], np.nan)
        prob_bck = np.where(consensus_bck[idx] == 1, prob_pred[idx], np.nan)

        mean_conf_organ = np.nanmean(prob_organ)
        mean_conf_bck = np.nanmean(prob_bck)
        confidence[organ_name] = ((1 - mean_conf_bck) + mean_conf_organ) / 2

        # Compute Dice score using MONAI
        dice_metric.reset()
        gt_tensor = torch.from_numpy(gt_masked)
        pred_tensor = torch.from_numpy(pred_masked)
        dice_metric(pred_tensor, gt_tensor)
        dice_scores[organ_name] = dice_metric.aggregate().item()

    return dice_scores, confidence

'''
Entropies evaluation
'''

def compute_entropy(image):
    """
    Calculate the entropy of an image (prediction or GT).
    
    Parameters:
    - image: numpy array representing the image or volume (slices, X, Y)
    """
    # Flatten the image and compute the histogram
    image_flat = image.flatten()
    
    # Calculate the histogram and normalize it
    hist, _ = np.histogram(image_flat, bins=256, range=(0, 256), density=True)
    
    # Compute the Shannon entropy
    entropy_value = entropy(hist)
    
    return entropy_value

'''
Hausdorff Distances evaluation
'''

def compute_hausdorff_distances (groundtruth, bin_pred):
    """
    Calculate the Hausdorff distance between the ground truth and prediction.
    
    Parameters:
    - gt: Ground truth binary segmentation 
    - pred: Predicted binary segmentation 
    
    @output hausdorff_distance
    """
    hausdorff_distances = {}
    organs = ['panc', 'kidn', 'livr']
    
    for i in range (3):

        # Extract the coordinates of the foreground (non-zero) pixels
        gt_coords = np.column_stack(np.where(groundtruth[i] > 0))
        pred_coords = np.column_stack(np.where(bin_pred > 0))
    
        # Calculate the directed Hausdorff distance in both directions
        forward_hausdorff = directed_hausdorff(gt_coords, pred_coords)[0]
        backward_hausdorff = directed_hausdorff(pred_coords, gt_coords)[0]
    
        # The Hausdorff distance is the maximum of the two directed distances
        hausdorff_distance = max(forward_hausdorff, backward_hausdorff)
        hausdorff_distances[i+1] = hausdorff_distance

    
    return hausdorff_distances

'''
Area Under the ROC Curve evaluation
'''
def compute_auroc(groundtruth, prob_pred):
    """
    Computes the AUROC (Area Under the Receiver Operating Characteristic Curve)
    for each class in a multi-class segmentation task.

    groundtruth: numpy array, shape (classes, slices, X, Y)
                 Ground truth segmentation masks (binary, 1 for presence of class, 0 for absence).
                 This should have dimensions: [3, slices, X, Y] where 3 is the number of classes (pancreas, kidney, liver).

    prob_pred: numpy array, shape (classes, slices, X, Y)
               The predicted probabilities for each class (output of softmax or sigmoid function).

    @output  auroc_dict
    """

    # Reshape groundtruth and prob_pred into 2D arrays for each class
    auroc_dict = {}
    organs =  {1: 'panc', 2: 'kidn', 3: 'livr'}
    num_classes = groundtruth.shape[0]  # Number of classes (3 for pancreas, kidney, liver)
    
    for i in range(num_classes):
        # Flatten the groundtruth for class i and the predicted probabilities for class i
        gt_class = (groundtruth[i] > 0).astype(np.uint8).flatten()  # Convert to binary groundtruth
        prob_class = prob_pred[i].flatten()  # Flatten the probabilities for class i
        
        # Calculate AUROC score for class i
        auroc_score = roc_auc_score(gt_class, prob_class)
        auroc_dict[organs[i + 1]] = auroc_score

    return auroc_dict

'''
Area Under the Risk Curve and Expected Area Under the Risk Curve evaluation
'''

@njit
def rc_curve_stats(risks: np.array, confids: np.array) -> tuple[list[float], list[float], list[float]]:
    coverages = []
    selective_risks = []
    assert len(risks.shape) == 1 and len(confids.shape) == 1 and len(risks) == len(confids)

    n_samples = len(risks)
    idx_sorted = np.argsort(confids)

    coverage = n_samples
    error_sum = np.sum(risks[idx_sorted])

    coverages.append(coverage / n_samples)
    selective_risks.append(error_sum / n_samples)

    weights = []
    tmp_weight = 0
    for i in range(0, len(idx_sorted) - 1):
        coverage = coverage - 1
        error_sum = error_sum - risks[idx_sorted[i]]
        tmp_weight += 1
        if i == 0 or confids[idx_sorted[i]] != confids[idx_sorted[i - 1]]:
            coverages.append(coverage / n_samples)
            selective_risks.append(error_sum / (n_samples - 1 - i))
            weights.append(tmp_weight / n_samples)
            tmp_weight = 0
    return coverages, selective_risks, weights

@njit
def calc_aurc(risks: np.array, confids: np.array):
    _, risks, weights = rc_curve_stats(risks, confids)
    return sum(
        [(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))]
    )

@njit
def calc_eaurc(risks: np.array, confids: np.array):
    """Compute normalized AURC, i.e. subtract AURC of optimal CSF (given fixed risks)."""
    n = len(risks)
    # optimal confidence sorts risk. Ascending here because we start from coverage 1/n
    selective_risks = np.sort(risks).cumsum() / np.arange(1, n + 1)
    aurc_opt = selective_risks.sum() / n
    return calc_aurc(risks, confids) - aurc_opt

def compute_aurc_eaurc(groundtruth, prob_pred):
    """
    Compute AURC and EAURC for a single class in the segmentation task.

    groundtruth: numpy array, shape (slices, X, Y)
                 Binary groundtruth segmentation mask for a specific class.

    prob_pred: numpy array, shape (num_classes, slices, X, Y)
               Predicted probabilities for each class (output of softmax/sigmoid).

    @output aurc_scores, eaurc_scores
    """
    aurc_scores = {}
    eaurc_scores = {}
    organs = ['panc', 'kidn', 'livr']

    for i, organ in enumerate(organs):

        # Flatten groundtruth and probabilities for the specific class
        gt_class = (groundtruth[i] > 0).astype(np.uint8).flatten()
        prob_class = prob_pred[i].flatten()

        # Compute risk (error) for each pixel (1 if incorrect, 0 if correct)
        risks = np.abs(gt_class - prob_class)  # Binary error (0 or 1)
        confids = prob_class  # Confidence is the predicted probability for the class

        # Calculate AURC and EAURC for this class
        class_aurc = calc_aurc(risks, confids)
        class_eaurc = calc_eaurc(risks, confids)
        aurc_scores[organ] = class_aurc
        eaurc_scores[organ] = class_eaurc

    return aurc_scores, eaurc_scores

'''
Expected Calibration Error evaluation
'''

def multirater_ece(annotations_list, prob_pred, device='cpu'):
    """
    Compute ECE per annotation (3 in total), optimized version.
    """
    with torch.no_grad():
        annotations_tensor = torch.tensor(np.stack(annotations_list), device=device)  # shape: (3, S, X, Y)
        prob_pred_tensor = torch.tensor(prob_pred, device=device)  # shape: (3, S, X, Y)

        ece_dict = {}

        for i in range(3):
            ece = calc_ece_optimized(annotations_tensor[i], prob_pred_tensor)
            ece_dict[i + 1] = ece

        return ece_dict

def calc_ece_optimized(groundtruth, prob_pred_onehot, num_classes=4, n_bins=50):
    """
    Optimized ECE computation using torch only, no autograd, no reshaping overheads.
    """
    with torch.no_grad():
        # Ensure shapes: (3, S, X, Y)
        background_prob = 1 - prob_pred_onehot.sum(dim=0, keepdim=True)  # shape: (1, S, X, Y)
        all_samples_with_bg = torch.cat((background_prob, prob_pred_onehot), dim=0)  # shape: (4, S, X, Y)

        # Flatten to (N, 4)
        all_samples_flat = all_samples_with_bg.permute(1, 2, 3, 0).reshape(-1, num_classes)  # (N, 4)
        all_groundtruth_flat = groundtruth.reshape(-1)  # (N,)

        # Metric (torchmetrics handles one-hot internally)
        calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins)
        ece = calibration_error(all_samples_flat, all_groundtruth_flat)

        return float(ece.cpu().numpy())

'''
Average Calibration Error evaluation    
'''

def prepare_inputs_for_ace(groundtruth, bin_pred, prob_pred):
    background_prob = 1 - np.sum(prob_pred, axis=0, keepdims=True)
    prob_pred_full = np.concatenate([background_prob, prob_pred], axis=0)

    confids = np.max(prob_pred_full, axis=0)
    flat_pred = bin_pred.flatten()
    flat_gt = groundtruth.flatten()
    flat_conf = confids.flatten()

    correct = (flat_pred == flat_gt).astype(np.int32)

    return correct, flat_conf

def calib_stats(correct, calib_confids):
    n_bins = 20
    y_true = column_or_1d(correct)
    y_prob = column_or_1d(calib_confids)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1]")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(f"Only binary classification is supported. Provided labels {labels}.")
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    num_nonzero = len(nonzero[nonzero == True])
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    bin_discrepancies = np.abs(prob_true - prob_pred)
    return bin_discrepancies, num_nonzero

def multirater_ace(annotations_list, bin_pred, prob_pred):

    ace_dict = {1: 0, 2: 0, 3: 0}
    for i in range(3):
        gt_i = annotations_list[i]
        correct, calib_confids = prepare_inputs_for_ace(gt_i, bin_pred, prob_pred)
        bin_discrepancies, num_nonzero = calib_stats(correct, calib_confids)
        ace_score = (1 / num_nonzero) * np.sum(bin_discrepancies)
        ace_dict[i+1] = ace_score
    return ace_dict

'''
CRPS  evaluation
'''

def volume_metric(groundtruth, prediction, voxel_proportion=1):
    cdf_dict, mean_gauss, var_gauss = calculate_volumes_distributions(groundtruth, voxel_proportion)

    crps_dict = {}
    organs = ['panc', 'kidn', 'livr']

    for i, organ_name in enumerate(organs):
        probabilistic_volume = compute_probabilistic_volume(prediction[i], voxel_proportion)
        crps_dict[organ_name] = crps_computation_fast(
            predicted_volume=probabilistic_volume,
            cdf=cdf_dict[organ_name],
            mean=mean_gauss[organ_name],
            std_dev=var_gauss[organ_name]
        )

    return crps_dict


def calculate_volumes_distributions(groundtruth, voxel_proportion=1):
    """
    From multiple GT annotations, compute volume stats and Gaussian CDF approximations.
    """
    organs = {1: 'panc', 2: 'kidn', 3: 'livr'}

    volumes = {}
    mean_gauss = {}
    var_gauss = {}

    for organ_val, organ_name in organs.items():
        vols = []
        for gt in groundtruth:
            count = np.sum(gt == organ_val)
            vols.append(count * np.prod(voxel_proportion))
        volumes[organ_name] = vols
        mean_gauss[organ_name] = np.mean(vols)
        var_gauss[organ_name] = np.std(vols)

    # Build CDFs using normal approximation (interpolated)
    cdfs = {}
    for organ_name in organs.values():
        mean = mean_gauss[organ_name]
        std = var_gauss[organ_name]
        x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
        cdf_vals = 0.5 * (1 + erf_vectorized((x - mean) / (std * np.sqrt(2))))
        cdfs[organ_name] = interp1d(x, cdf_vals, bounds_error=False, fill_value=(0.0, 1.0))

    return cdfs, mean_gauss, var_gauss


def compute_probabilistic_volume(preds, voxel_proportion=1):
    """
    Compute the soft volume from probabilistic segmentation (expected volume).
    """
    volume = preds.sum().item()
    return volume * voxel_proportion


def crps_computation_fast(predicted_volume, cdf, mean, std_dev, num_points=500):
    """
    CRPS approximation using trapezoidal integration and JIT-accelerated heaviside.
    """
    lower_limit = mean - 4 * std_dev
    upper_limit = mean + 4 * std_dev

    x = np.linspace(lower_limit, upper_limit, num_points)
    cdf_values = cdf(x)
    heaviside_values = heaviside_vectorized(x - predicted_volume)

    integrand = (cdf_values - heaviside_values) ** 2
    crps_value = np.trapz(integrand, x)

    return crps_value


@njit
def heaviside_vectorized(x):
    """
    Numba-accelerated Heaviside step function.
    """
    return 0.5 * (np.sign(x) + 1.0)

@njit
def erf_vectorized(x):
    """
    Numba-accelerated error function approximation.
    Equivalent to scipy.special.erf, needed for CDF of normal.
    """
    # Abramowitz and Stegun formula 7.1.26 for erf(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * np.exp(-x * x)

    return sign * y

'''
NCC
'''

def compute_ncc(groundtruth, prob_pred):
    """
    Compute the normalized cross correlation between a ground truth uncertainty and a predicted uncertainty map,
    to determine how similar the maps are.
    :param gt_unc_map: the ground truth uncertainty map based on the rater variability
    :param pred_unc_map: the predicted uncertainty map
    :return: float: the normalized cross correlation between gt and predicted uncertainty map
    """
    
    ncc_dict = {}
    organ_name = ['GT1-2', 'GT1-3', 'GT2-3']
    pairs = [(0, 1), (0, 2), (1, 2)]

    for idx, (i, j) in enumerate(pairs):
        gt1 = groundtruth[i].astype(np.float32)
        gt2 = groundtruth[j].astype(np.float32)

        mu_gt1 = np.mean(gt1)
        mu_gt2 = np.mean(gt2)
        sigma_gt1 = np.std(gt1, ddof=1)
        sigma_gt2 = np.std(gt2, ddof=1)

        gt1_norm = gt1 - mu_gt1
        gt2_norm = gt2 - mu_gt2
        prod = np.sum(gt1_norm * gt2_norm)

        ncc = (1 / (np.size(gt1) * sigma_gt1 * sigma_gt2)) * prod
        ncc_dict[organ_name[idx]] = ncc

    # Optionally: mean of the 3 NCCs
    ncc_dict['mean'] = np.mean(list(ncc_dict.values()))

    return ncc_dict

"""
Preprocessing functions : 
- files_to_data (extracts and converts data)
- preprocess_results (cropping)
"""

def compute_patient_crop_box(annotations):
    """
    Calcule la bounding box du foreground (zones annotées) sur toutes les annotations.
    """
    # Union des 3 annotations
    union_mask = np.sum(np.stack(annotations, axis=0), axis=0) > 0

    # Create a MONAI cropper
    cropper = CropForeground(select_fn=lambda x: x > 0, allow_smaller=True)
    box_start, box_end = cropper.compute_bounding_box(union_mask.astype(np.uint8))

    return box_start, box_end


def preprocess_results(ct_image, annotations, results, box_start, box_end, padding=0):
    """
    Utilise une bounding box prédéfinie pour croper toutes les images du patient.
    """
    box_start = [max(0, s - padding) for s in box_start]
    box_end = [min(ct_image.shape[1:][i], e + padding) for i, e in enumerate(box_end)]

    cropped_annotations = [a[..., box_start[0]:box_end[0], box_start[1]:box_end[1]] for a in annotations]
    cropped_results = [r[..., box_start[0]:box_end[0], box_start[1]:box_end[1]] for r in results]
    return cropped_annotations, cropped_results[0], cropped_results[1:]


def sitk_to_array(sitk_img):
    """ Convert SimpleITK image to numpy array with shape (slices, H, W) """
    array = sitk.GetArrayFromImage(sitk_img)  # Gives shape (slices, H, W)
    return array.astype(np.float32)


def getting_gt(gt_folder): 
    gt_files = sorted([f for f in os.listdir(gt_folder) if "annotation" in f])
    ct_file = next(f for f in os.listdir(gt_folder) if "image" in f)

    # Dowloading annotations (shape: (slices, H, W))
    annotations = []
    for f in gt_files:
        img = sitk.ReadImage(os.path.join(gt_folder, f))
        annotations.append(sitk_to_array(img))

    # CT scan
    ct_img = sitk.ReadImage(os.path.join(gt_folder, ct_file))
    ct_array = sitk_to_array(ct_img)
    return ct_array, annotations


def files_to_data(result_file, prob_file):
    # obtaining files

    # dowloading prediction from nnU-Net
    bin_pred_img = sitk.ReadImage(result_file)
    bin_pred = sitk_to_array(bin_pred_img).astype(np.uint8)

    # Dowloading probabilities (shape: (classes, slices, H, W)) and permutation 
    prob_data = np.load(prob_file)
    prob_array = prob_data[prob_data.files[0]]  # shape: (C, Z, H, W)
    # Extraction of the 3 classes in shape (slices, H, W)
    pancreas_conf = prob_array[1]
    kidney_conf = prob_array[2]
    liver_conf = prob_array[3]

    results = [bin_pred, pancreas_conf, kidney_conf, liver_conf]
    return results

'''
Applying the metrics to the inputed data
'''

from concurrent.futures import ProcessPoolExecutor

# === WRAPPERS FOR PARALLELIZATION ===
def compute_dice_parallel(args):
    return compute_consensus_dice_score(*args)

def compute_ece_parallel(args):
    return multirater_ece(*args)

def compute_crps_parallel(args):
    return volume_metric(*args)

def compute_auroc_parallel(args):
    return compute_auroc(*args)

def compute_aurc_eaurc_parallel(args):
    return compute_aurc_eaurc(*args)


def compute_fast_metrics_parallel(args):
    annotations_stack, cropped_annotations, cropped_bin_pred, cropped_prob_pred = args

    entropy_gt = compute_entropy(annotations_stack)
    entropy_pred = compute_entropy(cropped_bin_pred)
    hausdorff_distances = compute_hausdorff_distances(cropped_annotations, cropped_bin_pred)
    ace_dict = multirater_ace(cropped_annotations, cropped_bin_pred, cropped_prob_pred)
    ncc_dict = compute_ncc(cropped_annotations, cropped_prob_pred)

    return entropy_gt, entropy_pred, hausdorff_distances, ace_dict, ncc_dict


def apply_metrics(l_model_files, ct_image, annotations):
    '''
    Apply selected metrics with simple and clear parallelization.
    '''
    # Extract patient ID
    ct_name = re.findall(r"\/([^\/]+)$", l_patient_files["pred"])[0]

    # Load and preprocess data
    results = files_to_data(
        l_model_files["pred"], l_model_files["prob"]
    )
    box_start, box_end = compute_patient_crop_box(annotations)
    cropped_annotations, cropped_bin_pred, cropped_prob_pred = preprocess_results(
        ct_image, annotations, results, box_start, box_end
    )

    annotations_stack = np.stack(cropped_annotations, axis=0)

    # Prepare arguments
    args_dice = (annotations_stack, cropped_bin_pred, cropped_prob_pred)
    args_ece = (cropped_annotations, cropped_prob_pred)
    args_crps = (annotations_stack, cropped_prob_pred)
    args_aurc_eaurc = (annotations_stack, cropped_prob_pred)
    args_auroc = (annotations_stack, cropped_prob_pred)
    args_fast = (annotations_stack, cropped_annotations, cropped_bin_pred, cropped_prob_pred)

    # Parallel computation
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_dice = executor.submit(compute_dice_parallel, args_dice)
        future_ece = executor.submit(compute_ece_parallel, args_ece)
        future_crps = executor.submit(compute_crps_parallel, args_crps)
        future_aurc_eaurc = executor.submit(compute_aurc_eaurc_parallel, args_aurc_eaurc)
        future_auroc = executor.submit(compute_auroc_parallel, args_auroc)
        future_fast = executor.submit(compute_fast_metrics_parallel, args_fast)

        dice_scores, conf_scores = future_dice.result()
        print(f"[{ct_name}] DICE: {dice_scores} CONF : {conf_scores}")

        ece_scores = future_ece.result()
        print(f"[{ct_name}] ECE: {ece_scores}")

        crps_score = future_crps.result()
        print(f"[{ct_name}] CRPS: {crps_score}")

        auroc_score = future_auroc.result()
        print(f"[{ct_name}] AUROC: {auroc_score}")

        aurc_score, eaurc_score = future_aurc_eaurc.result()
        print(f"[{ct_name}] AURC: {aurc_score}")
        print(f"[{ct_name}] EAURC: {eaurc_score}")

        entropy_gt, entropy_pred, hausdorff_distances, ace_dict, ncc_dict = future_fast.result()
        print(f"[{ct_name}] Entropy_GT: {entropy_gt}, Entropy_Pred: {entropy_pred}")
        print(f"[{ct_name}] Hausdorff: {hausdorff_distances}")
        print(f"[{ct_name}] ACE: {ace_dict}")
        print(f"[{ct_name}] NCC: {ncc_dict}")

    # Return results
    return {
        "CT": ct_name,
        "DICE_panc": dice_scores["panc"],
        "DICE_kidn": dice_scores["kidn"],
        "DICE_livr": dice_scores["livr"],
        "CONF_panc": conf_scores["panc"],
        "CONF_kidn": conf_scores["kidn"],
        "CONF_livr": conf_scores["livr"],
        "Entropy_GT": entropy_gt,
        "Entropy_Pred": entropy_pred,
        "Hausdorff_panc": hausdorff_distances[1],
        "Hausdorff_kidn": hausdorff_distances[2],
        "Hausdorff_livr": hausdorff_distances[3],
        "ECE_1": ece_scores[1],
        "ECE_2": ece_scores[2],
        "ECE_3": ece_scores[3],
        "ACE_1": ace_dict[1],
        "ACE_2": ace_dict[2],
        "ACE_3": ace_dict[3],
        "AUROC_panc" : auroc_score["panc"],
        "AUROC_kidn": auroc_score["kidn"],
        "AUROC_livr": auroc_score["livr"],
        "AURC_panc" : aurc_score["panc"],
        "AURC_kidn": aurc_score["kidn"],
        "AURC_livr": aurc_score["livr"],
        "EAURC_panc" : eaurc_score["panc"],
        "EAURC_kidn": eaurc_score["kidn"],
        "EAURC_livr": eaurc_score["livr"],
        "CRPS_panc": crps_score["panc"],
        "CRPS_kidn": crps_score["kidn"],
        "CRPS_livr": crps_score["livr"],
        "NCC_GT1-2": ncc_dict["GT1-2"],
        "NCC_GT1-3": ncc_dict["GT1-3"],
        "NCC_GT2-3": ncc_dict["GT2-3"],
        "NCC_mean": ncc_dict["mean"]
    }




#BODY
"""
Gather the input locations and apply the metrics to all predictions.
The input folder should be looking like this : 
-- Folder   -- Model_01   -- pred_01.nii.gz
                            -- pred_prob_01.npz
                            -- GT_01                -- image.nii.gz
                                                    -- annotation_1.nii.gz
                                                    -- annotation_2.nii.gz
                                                    -- annotation_3.nii.gz
            -- Model_02   -- pred_02.nii.gz
            ...
            ...
            ...
            -- GT   -- image.nii.gz
                    -- annotation_1.nii.gz
                    -- annotation_2.nii.gz
                    -- annotation_3.nii.gz
"""
