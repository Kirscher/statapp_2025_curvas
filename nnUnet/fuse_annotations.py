import numpy as np
#pip install nibabel
import nibabel as nib

def fuse_annotations(annotations_paths, output_path):
    """Fusionne les annotations en faisant une moyenne arrondie."""
    annotations = [nib.load(path).get_fdata() for path in annotations_paths]
    
    # Rounded average
    fused_annotation = np.round(np.mean(annotations, axis=0)).astype(np.uint8)
    
    # Save files in NIfTI
    fused_nifti = nib.Nifti1Image(fused_annotation, affine=nib.load(annotations_paths[0]).affine)
    nib.save(fused_nifti, output_path)


'''annotations_files = ["annotation_1.nii.gz", "annotation_1.nii.gz", "annotation_1.nii.gz"]
output_fused = "fused_annotation.nii.gz"
fuse_annotations(annotations_files, output_fused)'''
