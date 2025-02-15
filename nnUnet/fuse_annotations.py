import s3fs
import os
import nibabel as nib
import numpy as np
import os
import s3fs
#import zipfile


s3 = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key = os.getenv("AWS_ACCESS_KEY_ID"), 
    secret = os.getenv("AWS_SECRET_ACCESS_KEY"),    
    token = os.getenv("AWS_SESSION_TOKEN"))

source_folder = f"leoacpr"
print(s3.ls(source_folder))
source_path = 's3://leoacpr/training_set.zip'

print(s3.ls("leoacpr/diffusion"))

import os
import s3fs
import nibabel as nib
import numpy as np
from io import BytesIO

# Connexion à MinIO
s3 = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    token=os.getenv("AWS_SESSION_TOKEN")
)

source_folder = "leoacpr/diffusion"
target_folder = "leoacpr/diffusion/nnunet_dataset/nnUNet_raw/labelsTr"

# Lister les dossiers UKCH
ukch_folders = [folder for folder in s3.ls(source_folder) if folder.startswith(f"{source_folder}/UKCH")]

for folder in ukch_folders:
    # Lister les fichiers NIfTI dans le dossier
    nii_files = [file for file in s3.ls(folder) if file.endswith('.nii.gz') and 'annotation' in file]
    
    if len(nii_files) != 3:
        print(f"Attention: {folder} contient {len(nii_files)} annotations au lieu de 3")
        continue
    
    # Charger les annotations
    annotations = []
    for file in nii_files:
        with s3.open(file, 'rb') as f:
            nii = nib.load(BytesIO(f.read()))
            annotations.append(nii.get_fdata())
    
    # Fusion par moyenne pixel
    merged_annotation = np.mean(annotations, axis=0)
    merged_nii = nib.Nifti1Image(merged_annotation, affine=nii.affine)
    
    # Enregistrer dans le dossier cible
    merged_filename = f"{os.path.basename(folder)}_merged_annotation.nii.gz"
    merged_path = f"{target_folder}/{merged_filename}"
    
    with BytesIO() as output:
        nib.save(merged_nii, output)
        output.seek(0)
        s3.upload(output, merged_path)
        print(f"Annotation fusionnée sauvegardée : {merged_path}")

print("Fusion terminée.")
















# Exemple d'utilisation
#input_dir = '/path/to/training_set'  # Répertoire des données d'entrée
#output_dir = '/path/to/output'  # Répertoire de sortie pour les annotations fusionnées

#os.makedirs(output_dir, exist_ok=True)
#merge_annotations(input_dir, output_dir)

'''