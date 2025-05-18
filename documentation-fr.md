# Documentation de l'Interface en Ligne de Commande StatApp

## Introduction
Ce document fournit une documentation complète pour l'interface en ligne de commande StatApp. La CLI StatApp est un outil de segmentation médicale qui gère l'incertitude et la variabilité entre les annotateurs. Elle est construite sur nnU-Net, un framework pour la segmentation d'images prêt à l'emploi.

La CLI fournit des commandes pour :
- Préparer des ensembles de données
- Entraîner des modèles
- Exécuter des prédictions
- Combiner plusieurs modèles
- Calculer des métriques
- Gérer les artefacts dans le stockage S3

## Aperçu des Commandes

### about
#### Description
Affiche des informations sur le projet, y compris son objectif et ses contributeurs.

#### Utilisation
```
statapp about
```

#### Paramètres
Aucun

### upload
#### Description
La commande upload fournit des fonctionnalités pour téléverser des répertoires locaux vers le stockage S3. Elle comprend trois sous-commandes :

##### upload_data
Téléverse un répertoire local vers le dossier de données S3 défini dans le fichier .env.

##### upload_model_artifacts
Téléverse un répertoire local vers le dossier S3 artifacts/model défini dans le fichier .env.

##### upload_preprocessing_artifacts
Téléverse un répertoire local vers le dossier S3 artifacts/preprocessing défini dans le fichier .env.

#### Utilisation
```
statapp upload upload-data <directory> [--verbose]
statapp upload upload-model-artifacts <directory> <modelfolder> [--verbose]
statapp upload upload-preprocessing-artifacts <directory> <preprocessingfolder> [--verbose]
```

#### Paramètres
| Sous-commande | Paramètre | Type | Description |
|------------|-----------|------|-------------|
| upload-data | directory | chaîne | Chemin du répertoire local à téléverser |
| upload-data | --verbose, -v | drapeau | Activer la sortie détaillée |
| upload-model-artifacts | directory | chaîne | Chemin du répertoire local à téléverser |
| upload-model-artifacts | modelfolder | chaîne | Nom du sous-dossier du modèle |
| upload-model-artifacts | --verbose, -v | drapeau | Activer la sortie détaillée |
| upload-preprocessing-artifacts | directory | chaîne | Chemin du répertoire local à téléverser |
| upload-preprocessing-artifacts | preprocessingfolder | chaîne | Nom du sous-dossier de prétraitement |
| upload-preprocessing-artifacts | --verbose, -v | drapeau | Activer la sortie détaillée |

### empty-artifacts
#### Description
Supprime tous les fichiers et dossiers du dossier d'artefacts S3 défini dans le fichier .env. Cette opération ne peut pas être annulée, donc à utiliser avec précaution.

#### Utilisation
```
statapp empty-artifacts [--verbose] [--confirm]
```

#### Paramètres
| Paramètre | Type | Description |
|-----------|------|-------------|
| --verbose, -v | drapeau | Activer la sortie détaillée |
| --confirm, -c | drapeau | Confirmer la suppression sans demander |

### empty-data
#### Description
Supprime tous les fichiers et dossiers du dossier de données S3 défini dans le fichier .env. Cette opération ne peut pas être annulée, donc à utiliser avec précaution.

#### Utilisation
```
statapp empty-data [--verbose] [--confirm]
```

#### Paramètres
| Paramètre | Type | Description |
|-----------|------|-------------|
| --verbose, -v | drapeau | Activer la sortie détaillée |
| --confirm, -c | drapeau | Confirmer la suppression sans demander |

### prepare
#### Description
La commande prepare fournit des fonctionnalités pour préparer des ensembles de données pour l'analyse. Elle comprend trois sous-commandes :

##### download-dataset
Télécharge un ensemble de données pour analyse sans exécuter de prétraitement.

##### download-preprocessing
Télécharge des artefacts de prétraitement pour un ensemble de données.

##### prepare
Prépare un ensemble de données pour l'analyse en téléchargeant les données et en exécutant le prétraitement.

#### Utilisation
```
statapp prepare download-dataset <annotator> <patients> [--verbose]
statapp prepare download-preprocessing <annotator> <patients> [--verbose]
statapp prepare prepare <annotator> <patients> [--skip] [--num-processes-fingerprint <num>] [--num-processes <num>] [--verbose]
```

#### Paramètres
| Sous-commande | Paramètre | Type | Description |
|------------|-----------|------|-------------|
| download-dataset | annotator | chaîne | Annotateur (1/2/3) |
| download-dataset | patients | liste/chaîne | Liste de numéros de patients (ex. 001 034) ou 'all', 'train', 'validation', 'test' |
| download-dataset | --verbose, -v | drapeau | Activer la journalisation détaillée |
| download-preprocessing | annotator | chaîne | Annotateur (1/2/3) |
| download-preprocessing | patients | liste/chaîne | Liste de numéros de patients (ex. 001 034) ou 'all', 'train', 'validation', 'test' |
| download-preprocessing | --verbose, -v | drapeau | Activer la journalisation détaillée |
| prepare | annotator | chaîne | Annotateur (1/2/3) |
| prepare | patients | liste/chaîne | Liste de numéros de patients (ex. 001 034) ou 'all', 'train', 'validation', 'test' |
| prepare | --skip | drapeau | Ignorer le téléchargement et exécuter uniquement le prétraitement |
| prepare | --num-processes-fingerprint, -npfp | entier | Nombre de processus à utiliser pour l'extraction d'empreintes (par défaut : 2) |
| prepare | --num-processes, -np | entier | Nombre de processus à utiliser pour le prétraitement (par défaut : 2) |
| prepare | --verbose, -v | drapeau | Activer la journalisation détaillée |

### train
#### Description
Exécute l'entraînement nnUNet. L'ensemble de données doit être préparé avec la commande prepare au préalable.

#### Utilisation
```
statapp train [<seed>] [--fold <fold>] [--patients <patients>] [--annotator <annotator>] [--verbose]
```

#### Paramètres
| Paramètre | Type | Description |
|-----------|------|-------------|
| seed | entier | Définir la graine aléatoire pour la reproductibilité |
| --fold, -f | chaîne | Pli à utiliser pour l'entraînement. Peut être 'all' pour utiliser tous les plis, ou un numéro de pli spécifique (0-4) |
| --patients, -p | liste/chaîne | Liste de numéros de patients (ex. 001 034) ou 'all', 'train', 'validation', 'test' |
| --annotator, -a | chaîne | Annotateur (1/2/3) |
| --verbose, -v | drapeau | Activer la journalisation détaillée |

### run
#### Description
Exécute le pipeline complet : préparer les données, entraîner le modèle et téléverser les artefacts. Cette commande combine les fonctionnalités des commandes prepare, train et upload_artifacts.

#### Utilisation
```
statapp run <annotator> <seed> <patients> [--fold <fold>] [--skip] [--num-processes-fingerprint <num>] [--num-processes <num>] [--verbose]
```

#### Paramètres
| Paramètre | Type | Description |
|-----------|------|-------------|
| annotator | chaîne | Annotateur (1/2/3) |
| seed | entier | Définir la graine aléatoire pour la reproductibilité |
| patients | liste/chaîne | Liste de numéros de patients (ex. 001 034) ou 'all', 'train', 'validation', 'test' |
| --fold, -f | chaîne | Pli à utiliser pour l'entraînement. Peut être 'all' pour utiliser tous les plis, ou un numéro de pli spécifique (0-4) |
| --skip | drapeau | Ignorer le téléchargement et exécuter uniquement le prétraitement |
| --num-processes-fingerprint, -npfp | entier | Nombre de processus à utiliser pour l'extraction d'empreintes (par défaut : 2) |
| --num-processes, -np | entier | Nombre de processus à utiliser pour le prétraitement (par défaut : 2) |
| --verbose, -v | drapeau | Activer la journalisation détaillée |

### predict
#### Description
Prédit la segmentation pour les patients en utilisant les modèles spécifiés. Télécharge les images des patients et les points de contrôle des modèles, exécute la prédiction pour chaque modèle et téléverse les résultats vers S3.

#### Utilisation
```
statapp predict <patients> [--models <models>] [--jobs <num>] [--verbose]
```

#### Paramètres
| Paramètre | Type | Description |
|-----------|------|-------------|
| patients | liste/chaîne | Liste de numéros de patients (ex. 001 034) ou 'all', 'train', 'validation', 'test' |
| --models, -m | liste/chaîne | Liste de modèles à utiliser pour la prédiction (ex. anno1_init112233_foldall) ou 'all' |
| --jobs, -j | entier | Nombre de processus à exécuter (par défaut : 10) |
| --verbose, -v | drapeau | Activer la journalisation détaillée |

### ensemble
#### Description
La commande ensemble fournit des fonctionnalités pour combiner les prédictions de plusieurs modèles. Elle comprend trois sous-commandes :

##### dl-ensemble
Télécharge les prédictions de plusieurs modèles.

##### run-ensemble
Exécute ensemble_folders de nnUNet sur les prédictions de modèles téléchargées.

##### ensemble
Combine les fonctionnalités des commandes dl-ensemble et run-ensemble.

#### Utilisation
```
statapp ensemble dl-ensemble <patients> [--models <models>] [--jobs <num>] [--verbose]
statapp ensemble run-ensemble <patients> [--verbose] [--jobs <num>]
statapp ensemble ensemble <patients> [--models <models>] [--jobs <num>] [--verbose]
```

#### Paramètres
| Sous-commande | Paramètre | Type | Description |
|------------|-----------|------|-------------|
| dl-ensemble | patients | liste/chaîne | Liste de numéros de patients (ex. 001 034) ou 'all', 'train', 'validation', 'test' |
| dl-ensemble | --models, -m | liste/chaîne | Liste de modèles à utiliser pour l'ensemble (ex. anno1_init112233_foldall) ou 'all' |
| dl-ensemble | --jobs, -j | entier | Nombre de processus à exécuter (par défaut : 10) |
| dl-ensemble | --verbose, -v | drapeau | Activer la journalisation détaillée |
| run-ensemble | patients | liste/chaîne | Liste de numéros de patients (ex. 001 034) |
| run-ensemble | --jobs, -j | entier | Nombre de processus à exécuter (par défaut : 10) |
| run-ensemble | --verbose, -v | drapeau | Activer la journalisation détaillée |
| ensemble | patients | liste/chaîne | Liste de numéros de patients (ex. 001 034) ou 'all', 'train', 'validation', 'test' |
| ensemble | --models, -m | liste/chaîne | Liste de modèles à utiliser pour l'ensemble (ex. anno1_init112233_foldall) ou 'all' |
| ensemble | --jobs, -j | entier | Nombre de processus à exécuter (par défaut : 10) |
| ensemble | --verbose, -v | drapeau | Activer la journalisation détaillée |

### metrics
#### Description
La commande metrics fournit des fonctionnalités pour calculer et télécharger des métriques pour les prédictions de modèles. Elle comprend deux sous-commandes :

##### compute-metrics
Calcule des métriques pour les prédictions de modèles sur les données des patients.

##### dl-metrics
Télécharge tous les fichiers de métriques de S3, les fusionne et les enregistre dans le répertoire de travail.

#### Utilisation
```
statapp metrics compute-metrics <patients> [--models <models>] [--verbose]
statapp metrics dl-metrics [--output <output_name>] [--verbose]
```

#### Paramètres
| Sous-commande | Paramètre | Type | Description |
|------------|-----------|------|-------------|
| compute-metrics | patients | liste/chaîne | Liste de numéros de patients (ex. 075 034) ou 'all', 'train', 'validation', 'test' |
| compute-metrics | --models, -m | liste/chaîne | Liste de modèles à utiliser pour le calcul des métriques (ex. anno1_init112233_foldall) ou 'all' |
| compute-metrics | --verbose, -v | drapeau | Activer la journalisation détaillée |
| dl-metrics | --output, -o | chaîne | Nom du fichier CSV de sortie (par défaut : metrics.csv) |
| dl-metrics | --verbose, -v | drapeau | Activer la journalisation détaillée |

## Exemples

### Préparation d'un Ensemble de Données
```
# Télécharger un ensemble de données pour l'annotateur 1 avec tous les patients
statapp prepare download-dataset 1 all --verbose

# Télécharger des artefacts de prétraitement pour l'annotateur 1 avec les patients d'entraînement
statapp prepare download-preprocessing 1 train --verbose

# Préparer un ensemble de données pour l'annotateur 1 avec les patients de validation
statapp prepare prepare 1 validation --num-processes 4 --verbose
```

### Entraînement d'un Modèle
```
# Entraîner un modèle avec l'annotateur 1, la graine 42 et tous les plis
statapp train 42 --annotator 1 --fold all --patients train --verbose

# Exécuter le pipeline complet pour l'annotateur 1, la graine 42 et les patients de validation
statapp run 1 42 validation --fold all --num-processes 4 --verbose
```

### Exécution de Prédictions
```
# Prédire la segmentation pour les patients de test en utilisant tous les modèles
statapp predict test --models all --jobs 8 --verbose

# Prédire la segmentation pour des patients spécifiques en utilisant un modèle spécifique
statapp predict 001 034 --models anno1_init42_foldall --jobs 4 --verbose
```

### Combinaison de Modèles
```
# Télécharger les prédictions pour les patients de test à partir de tous les modèles
statapp ensemble dl-ensemble test --models all --jobs 8 --verbose

# Exécuter l'ensemble pour des patients spécifiques
statapp ensemble run-ensemble 001 034 --jobs 4 --verbose

# Exécuter le pipeline d'ensemble complet pour les patients de test
statapp ensemble ensemble test --models all --jobs 8 --verbose
```

### Calcul de Métriques
```
# Calculer des métriques pour les patients de test en utilisant tous les modèles
statapp metrics compute-metrics test --models all --verbose

# Télécharger et fusionner tous les fichiers de métriques
statapp metrics dl-metrics --output metriques_combinees.csv --verbose
```

### Gestion du Stockage S3
```
# Téléverser des données vers S3
statapp upload upload-data ./mes_donnees --verbose

# Téléverser des artefacts de modèle vers S3
statapp upload upload-model-artifacts ./mon_modele anno1_init42_foldall --verbose

# Vider le répertoire d'artefacts (avec confirmation)
statapp empty-artifacts --verbose

# Vider le répertoire de données (sans confirmation)
statapp empty-data --confirm --verbose
```

## Variables d'Environnement
La CLI StatApp utilise plusieurs variables d'environnement pour la configuration. Celles-ci doivent être définies dans un fichier .env dans le répertoire racine du projet.

| Variable | Description |
|----------|-------------|
| S3_BUCKET | Le nom du bucket S3 |
| S3_DATA_DIR | Le répertoire dans S3 pour stocker les données |
| S3_ARTIFACTS_DIR | Le répertoire dans S3 pour stocker les artefacts |
| S3_OUTPUT_DIR | Le répertoire dans S3 pour stocker la sortie |
| S3_METRICS_DIR | Le répertoire dans S3 pour stocker les métriques |
| S3_MODEL_ARTIFACTS_SUBDIR | Le sous-répertoire pour les artefacts de modèle (par défaut : models) |
| S3_PROPROCESSING_ARTIFACTS_SUBDIR | Le sous-répertoire pour les artefacts de prétraitement (par défaut : preprocessing) |
| SEED | Graine aléatoire pour la reproductibilité |