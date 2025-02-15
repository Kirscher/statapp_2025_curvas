import os
import nibabel as nib
import numpy as np
from pathlib import Path
import tempfile
import s3fs


# Connexion à MinIO S3 Onyxia
s3 = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    token=os.getenv("AWS_SESSION_TOKEN")
)

source_folder = "leoacpr/diffusion"
target_folder = "leoacpr/diffusion/nnunet_dataset/nnUNet_raw/labelsTr"

def merge_annotations(s3, source_folder="leoacpr/diffusion"):
    """
    Fusionne les annotations de CT scans au format nii.gz en prenant la moyenne
    et les sauvegarde dans le format attendu par nnU-Net.
    
    Args:
        s3: Instance s3fs.S3FileSystem 
        source_folder: Chemin du dossier source contenant les dossiers UKCH
    """
    # Liste tous les dossiers commençant par UKCH
    all_folders = [f for f in s3.ls(source_folder) if 'UKCH' in f]
    
    # Dossier de destination pour nnU-Net
    dest_folder = "leoacpr/diffusion/nnunet_dataset/nnUNet_raw/labelsTr"
    
    for folder in all_folders:
        # Liste les fichiers d'annotation dans le dossier
        files = [f for f in s3.ls(folder) if f.endswith('.nii.gz') and 'annotation' in f]
        
        if len(files) != 3:
            print(f"Attention: {folder} ne contient pas exactement 3 annotations")
            continue
            
        # Utilise un dossier temporaire pour les opérations sur les fichiers
        with tempfile.TemporaryDirectory() as temp_dir:
            # Charge les trois annotations
            annotations = []
            for file in files:
                temp_file = os.path.join(temp_dir, os.path.basename(file))
                s3.get(file, temp_file)
                img = nib.load(temp_file)
                annotations.append(img.get_fdata())
            
            merged_data = np.mean(annotations, axis=0)
            
            # Crée le nouveau fichier nii.gz avec la même affine et header que le premier
            merged_nii = nib.Nifti1Image(merged_data, img.affine, img.header)
            
            # Nom du fichier de sortie
            folder_name = os.path.basename(folder)
            output_filename = f"{folder_name}_merged_annotation.nii.gz"
            temp_output = os.path.join(temp_dir, output_filename)
            
            # Sauvegarde localement puis upload vers S3
            nib.save(merged_nii, temp_output)
            dest_path = f"{dest_folder}/{output_filename}"
            s3.put(temp_output, dest_path)
            
            print(f"Fusion terminée pour {folder_name}")

#merge_annotations(s3, source_folder="leoacpr/diffusion")

