echo "=== Réglage des variables d'environnement ==="

export nnUNet_raw="/home/onyxia/work/statapp_2025_curvas/nnUNet_raw"
export nnUNet_preprocessed="/home/onyxia/work/statapp_2025_curvas/nnUNet_preprocessed"
export nnUNet_results="/home/onyxia/work/statapp_2025_curvas/nnUNet_results"

echo "=== Pré-processing nnUNet (475_CURVAS) ==="

nnUNetv2_plan_and_preprocess -d 475 --verify_dataset_integrity -c 3d_fullres

echo "=== Entraînement NNUnet (475_CURVAS) ==="

nnUNetv2_train 475 3d_fullres 0 --npz --c