# Training nnUnet

# to download later:
#(OPTIONAL) Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network 
# topologies it generates (see Model training). To install hiddenlayer, run the following command:

from tqdm import tqdm
import torch
#pip3 install nnunetv2
import s3fs
import os
from pathlib import Path


# Connexion à MinIO S3 Onyxia
s3 = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    token=os.getenv("AWS_SESSION_TOKEN")
)

print(len(s3.ls("projet-statapp-segmedic/diffusion/nnunet_dataset/nnUNet_raw/Dataset001_Annot1/labelsTr")))



def download_s3_folder():
    
    # Définir les chemins
    base_local_path = Path('/tmp/nnunet')
    s3_base_path = "projet-statapp-segmedic/diffusion/nnunet_dataset"
    folders = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    
    # Créer les dossiers locaux
    for folder in folders:
        local_folder = base_local_path / folder
        local_folder.mkdir(parents=True, exist_ok=True)
        
        # Chemin S3 complet
        s3_path = f"{s3_base_path}/{folder}"
        print(f"\nTéléchargement du dossier {folder}...")
        
        # Lister récursivement tous les fichiers
        try:
            files = s3.find(s3_path)
            
            # Créer une barre de progression
            with tqdm(total=len(files), desc=f"Fichiers dans {folder}") as pbar:
                for file_path in files:
                    # Calculer le chemin local relatif
                    relative_path = file_path.replace(s3_path, '').lstrip('/')
                    local_file_path = local_folder / relative_path
                    
                    # Créer les sous-dossiers si nécessaire
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Télécharger le fichier
                    if not local_file_path.exists():
                        try:
                            s3.get(file_path, str(local_file_path))
                        except Exception as e:
                            print(f"Erreur lors du téléchargement de {file_path}: {e}")
                    
                    pbar.update(1)
        
        except Exception as e:
            print(f"Erreur lors de la lecture du dossier {s3_path}: {e}")
            continue
        
        #ERROR CORRECTED: the nnU-Net dataset naming convention requires 4 digit for image case file, not 3. 
        for string in ['1', '2', '3']:
            images = Path(f"/tmp/nnunet/nnUNet_raw/Dataset001_Annot{string}/imagesTr")
            for f in images.glob("*_000.nii.gz"):
                f.rename(f.with_name(f.name.replace("_000.nii.gz", "_0000.nii.gz")))
    
    # Configurer les variables d'environnement
    env_vars = {
        'nnUNet_raw': str(base_local_path / 'nnUNet_raw'),
        'nnUNet_preprocessed': str(base_local_path / 'nnUNet_preprocessed'),
        'nnUNet_results': str(base_local_path / 'nnUNet_results')
    }
    
    # Mettre à jour les variables d'environnement
    for var_name, path in env_vars.items():
        os.environ[var_name] = path
    
    # Ajouter au .bashrc
    with open(os.path.expanduser('~/.bashrc'), 'a') as f:
        f.write('\n# nnUNet paths\n')
        for var_name, path in env_vars.items():
            f.write(f'export {var_name}="{path}"\n')
    
    print("\nConfiguration terminée. Variables d'environnement définies :")
    for var_name, path in env_vars.items():
        print(f"{var_name}={path}")
    
    print("\nRedémarrez votre shell ou exécutez 'source ~/.bashrc' pour appliquer les changements.")

"""if __name__ == "__main__":
    download_s3_folder()"""

# il faudra uploader les documents dans le S3 onyxia attention ! car sinon tout est chargé en local"
# tâche : reprendre tout le code et créer une fonction qui prend comme paramètre le Dataset d'annotation ciblé (on s'occupera des initialisations plus tard). 
# en faire un jupyter notebook

def upload_preprocessed_to_s3():
    from pathlib import Path
    from tqdm import tqdm

    # Dossier local et distant
    local_folder = Path('/tmp/nnunet/nnUNet_preprocessed')
    s3_folder = "projet-statapp-segmedic/diffusion/nnunet_dataset/nnUNet_preprocessed"
    
    # Lister tous les fichiers à uploader
    files = list(local_folder.rglob("*"))
    
    print(f"\nUploading nnUNet_preprocessed to {s3_folder}...")
    with tqdm(total=len(files), desc="Upload nnUNet_preprocessed") as pbar:
        for file_path in files:
            if file_path.is_file():
                relative_path = file_path.relative_to(local_folder)
                s3_path = f"{s3_folder}/{relative_path.as_posix()}"
                try:
                    s3.put(str(file_path), s3_path)
                except Exception as e:
                    print(f"Erreur lors de l'upload de {file_path} → {s3_path}: {e}")
            pbar.update(1)

# Appel si ce fichier est exécuté directement
"""if __name__ == "__main__":
    upload_preprocessed_to_s3()"""




## Code to train the first neural network (Dataset001_Annot1)

# Strategy: using multithreading to run the training and the upload to S3. 

import threading
import subprocess
import time

local_results_path = Path("/tmp/nnunet/nnUNet_results/Dataset001_Annot1")
s3_results_path = "projet-statapp-segmedic/diffusion/nnunet_dataset/nnUNet_results/Dataset001_Annot1"

# Upload function with time interval = 180 (could be longer maybe)
# more smartly: upload as soon as the content of temp/results changes
def sync_results_to_s3(interval=180):
    print("[Uploader] Starting S3 sync thread.")
    uploaded_files = set()
    while True:
        for file_path in local_results_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_results_path)
                s3_path = f"{s3_results_path}/{rel_path.as_posix()}"
                if s3_path not in uploaded_files:
                    try:
                        s3.put(str(file_path), s3_path)
                        uploaded_files.add(s3_path)
                        print(f"[Uploader] Uploaded: {s3_path}")
                    except Exception as e:
                        print(f"[Uploader] Error uploading {s3_path}: {e}")
        #time.sleep(interval)


# Training function
def run_training():
    print("[Trainer] Launching nnUNet training...")
    command = [
        "nnUNetv2_train",
        "Dataset002_Annot2",  # Dataset ID
        "3d_fullres",  # Plan
        "all",  # Fold            to try with all folds, need for-loop
        "--npz"
    ]
    subprocess.run(command)
    print("[Trainer] Training complete.")


"""
# Threads
uploader_thread = threading.Thread(target=sync_results_to_s3, daemon=True)
trainer_thread = threading.Thread(target=run_training)

uploader_thread.start()
trainer_thread.start()

trainer_thread.join()
print("[Main] All done.")"""
