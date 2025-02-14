import s3fs
import os
import nibabel as nib
import numpy as np
import os
import s3fs
import zipfile


s3 = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key = os.getenv("AWS_ACCESS_KEY_ID"), 
    secret = os.getenv("AWS_SECRET_ACCESS_KEY"),    
    token = os.getenv("AWS_SESSION_TOKEN"))

source_folder = f"leoacpr"
print(s3.ls(source_folder))
source_path = 's3://leoacpr/training_set.zip'







# Définir les chemins source et destination
bucket_name = 'leoacpr'
zip_file_key = 'diffusion/training_set.zip'
destination_folder = 'diffusion/'

# Télécharger le fichier ZIP depuis S3
zip_file_path = f's3://{bucket_name}/{zip_file_key}'
zip_file = s3.open(zip_file_path, 'rb')

# Fonction pour fusionner les annotations
def merge_annotations(zip_file, output_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Liste des sous-dossiers (cas patients)
        cases = [name for name in zip_ref.namelist() if name.endswith('/') and 'UKCHLL' in name]

        for case in cases:
            annotation_files = [name for name in zip_ref.namelist() if name.startswith(case) and 'annotation' in name]

            # Charger les annotations
            annotations = []
            for annotation_file in annotation_files:
                with zip_ref.open(annotation_file) as file:
                    annotation_img = nib.load(io.BytesIO(file.read()))
                    annotations.append(annotation_img.get_fdata())

            # Calculer la moyenne des annotations
            merged_annotation = np.mean(annotations, axis=0)

            # Sauvegarder l'annotation fusionnée dans le même dossier que les annotations d'origine
            merged_annotation_img = nib.Nifti1Image(merged_annotation, annotation_img.affine)
            merged_annotation_path = os.path.join(case, 'merged_annotation.nii.gz')

            # Sauvegarder l'annotation fusionnée dans le fichier ZIP
            with io.BytesIO() as merged_annotation_buffer:
                nib.save(merged_annotation_img, merged_annotation_buffer)
                merged_annotation_buffer.seek(0)
                with s3.open(f's3://{bucket_name}/{merged_annotation_path}', 'wb') as s3_file:
                    s3_file.write(merged_annotation_buffer.read())

            print(f"Fusionnée {case}: s3://{bucket_name}/{merged_annotation_path}")

# Appeler la fonction pour fusionner les annotations
merge_annotations(zip_file, destination_folder)

print("Fusion des annotations terminée.")













# Exemple d'utilisation
'''input_dir = '/path/to/training_set'  # Répertoire des données d'entrée
output_dir = '/path/to/output'  # Répertoire de sortie pour les annotations fusionnées

os.makedirs(output_dir, exist_ok=True)
merge_annotations(input_dir, output_dir)'''

