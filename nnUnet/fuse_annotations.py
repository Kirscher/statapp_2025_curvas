#utiliser S3 cf mon projet de data science

import os
import nibabel as nib
import numpy as np

def merge_annotations(input_dir, output_dir):
    # Liste des sous-dossiers (cas patients)
    cases = [case for case in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, case))]
    
    for case in cases:
        case_dir = os.path.join(input_dir, case)
        annotation_files = [f for f in os.listdir(case_dir) if f.startswith('annotation')]
        
        # Charger les annotations
        annotations = []
        for annotation_file in annotation_files:
            annotation_path = os.path.join(case_dir, annotation_file)
            annotation_img = nib.load(annotation_path)
            annotations.append(annotation_img.get_fdata())
        
        # Calculer la moyenne des annotations
        merged_annotation = np.mean(annotations, axis=0)
        
        # Sauvegarder l'annotation fusionnée
        merged_annotation_img = nib.Nifti1Image(merged_annotation, annotation_img.affine)
        output_path = os.path.join(output_dir, case + '_merged_annotation.nii.gz')
        nib.save(merged_annotation_img, output_path)
        print(f"Fusionnée {case}: {output_path}")

# Exemple d'utilisation
input_dir = '/path/to/training_set'  # Répertoire des données d'entrée
output_dir = '/path/to/output'  # Répertoire de sortie pour les annotations fusionnées

os.makedirs(output_dir, exist_ok=True)
merge_annotations(input_dir, output_dir)

