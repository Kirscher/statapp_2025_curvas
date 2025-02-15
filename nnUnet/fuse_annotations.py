import os
import s3fs
import nibabel as nib
import numpy as np
from io import BytesIO
from nibabel.filebasedimages import FileHolder

# Connexion à MinIO
s3 = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://' + 'minio.lab.sspcloud.fr'},
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    token=os.getenv("AWS_SESSION_TOKEN")
)

source_folder = "leoacpr/diffusion"
target_folder = "leoacpr/diffusion/nnunet_dataset/nnUNet_raw/labelsTr"

ukch_folders = [folder for folder in s3.ls(source_folder) if folder.startswith(f"{source_folder}/UKCH")]

for folder in ukch_folders:
    nii_files = [file for file in s3.ls(folder) if file.endswith('.nii.gz') and 'annotation' in file]
    if len(nii_files) != 3:
        print(f"Attention: {folder} contient {len(nii_files)} annotations au lieu de 3")
        continue

    annotations = []
    affine = None
    for file in nii_files:
        with s3.open(file, 'rb') as f:
            try:
                file_content = f.read()
                file_like = BytesIO(file_content)
                file_holder = FileHolder(fileobj=file_like)
                nii = nib.Nifti1Image.from_file_map({'header': file_holder, 'image': file_holder})
                if affine is None:
                    affine = nii.affine
                annotations.append(nii.get_fdata())
            except nib.spatialimages.HeaderDataError:
                print(f"Problème de header avec {file}, tentative de chargement forcé.")
                try:
                    # Forcer le chargement en ignorant les erreurs d'en-tête
                    nii = nib.Nifti1Image.from_file_map({'header': file_holder, 'image': file_holder})
                    if affine is None:
                        affine = nii.affine
                    annotations.append(nii.get_fdata())
                except Exception as e:
                    print(f"Erreur lors du chargement forcé de {file}: {e}")
                    continue
            except Exception as e:
                print(f"Erreur lors du chargement de {file}: {e}")
                continue

    if not annotations:
        print(f"Aucune annotation valide chargée pour {folder}.")
        continue

    merged_annotation = np.mean(annotations, axis=0).astype(np.float32)
    merged_nii = nib.Nifti1Image(merged_annotation, affine=affine)
    merged_nii.header.set_data_dtype(np.float32)

    merged_filename = f"{os.path.basename(folder)}_merged_annotation.nii.gz"
    merged_path = f"{target_folder}/{merged_filename}"

    with BytesIO() as output:
        nib.save(merged_nii, output)
        output.seek(0)
        with s3.open(merged_path, 'wb') as out_file:
            out_file.write(output.read())
        print(f"Annotation fusionnée sauvegardée : {merged_path}")

print("Fusion terminée.")
