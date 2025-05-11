

import os
os.environ['nnUNet_results'] = '/home/onyxia/statapp_2025_curvas/nnUNet/nnUNet_results'
os.environ['nnUNet_raw'] = '/home/onyxia/statapp_2025_curvas/nnUNet/nnUNet_raw'
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO 
import nibabel as nib
import numpy as np

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    verbose=True,
    verbose_preprocessing=False,
    allow_tqdm=True
)
predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'nnUNet/model'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    
img, props = SimpleITKIO().read_images(["/home/onyxia/statapp_2025_curvas/nnUNet/nnUNet_raw/Dataset001_test/UKCHLL061/image.nii.gz"])
ret = predictor.predict_single_npy_array(img, props, None, "please", save_or_return_probabilities=True)
"""predictor.predict_from_files(join(nnUNet_raw, 'Dataset001_test/UKCHLL059'),
                                 join(nnUNet_raw, 'feur'),
                                 save_probabilities=True, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)"""
# predict a single numpy array (NibabelIO)
#img, props = NibabelIO().read_images("/home/onyxia/statapp_2025_curvas/test/UKCHLL058/image.nii.gz")
#ret = predictor.predict_single_npy_array(img, props, None, None, False)

# The following IS NOT RECOMMENDED. Use nnunetv2.imageio!
# nibabel, we need to transpose axes and spacing to match the training axes ordering for the nnU-Net default:
"""nib.load("/home/onyxia/statapp_2025_curvas/test/UKCHLL058/image.nii.gz")
img = np.asanyarray(img_nii.dataobj).transpose([2, 1, 0])  # reverse axis order to match SITK
props = {'spacing': img_nii.header.get_zooms()[::-1]}      # reverse axis order to match SITK
ret = predictor.predict_single_npy_array(img, props, None, None, False)"""