# Training nnUnet

# to download later:
#(OPTIONAL) Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network 
# topologies it generates (see Model training). To install hiddenlayer, run the following command:

import torch
#pip3 install nnunetv2
import s3fs
import os
from pathlib import Path
from tqdm import tqdm

# Connexion à MinIO S3 Onyxia
s3 = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    token=os.getenv("AWS_SESSION_TOKEN")
)

print(len(s3.ls("leoacpr/diffusion/nnunet_dataset/nnUNet_raw/Dataset001_finetune/labelsTr")))



def download_s3_folder():
    
    # Définir les chemins
    base_local_path = Path('/tmp/nnunet')
    s3_base_path = "leoacpr/diffusion/nnunet_dataset"
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

if __name__ == "__main__":
    download_s3_folder()
