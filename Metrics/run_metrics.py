import os
from Metrics_func.new_metrics import apply_metrics



#BODY
"""
Gather the input locations and apply the metrics to all predictions.
The input folder should be looking like this : 
-- Folder   -- Patient_01   -- pred_01.nii.gz
                            -- pred_prob_01.npz
                            -- GT_01                -- image.nii.gz
                                                    -- annotation_1.nii.gz
                                                    -- annotation_2.nii.gz
                                                    -- annotation_3.nii.gz
            -- Patient_02   -- pred_02.nii.gz
            ...
            ...
            ...
"""

#Locating data
data_path = "/home/onyxia/statapp_2025_curvas/truc"#input(str("Path to the folder with all the patient folders : "))
l_patients_path = os.listdir(data_path)
l_patients=[]

for patient_dir in l_patients_path:
    patient_dict = {"pred": None, "prob": None, "gt": None}
    patient_path = os.path.join(data_path, patient_dir)
    for item in os.listdir(patient_path):
            item_path = os.path.join(patient_path, item)
            if item.endswith(".nii.gz"):
                patient_dict["pred"] = item_path
            elif item.endswith(".npz"):
                patient_dict["prob"] = item_path
            elif os.path.isdir(item_path):
                patient_dict["gt"] = item_path

    l_patients.append(patient_dict)

#Prepare output
df = pd.DataFrame(columns=[
        "CT",
        "DICE_panc",
        "DICE_kidn",
        "DICE_livr",
        "CONF_panc",
        "CONF_kidn",
        "CONF_livr",
        "Entropy_GT",
        "Entropy_Pred",
        "Hausdorff_1",
        "Hausdorff_2",
        "Hausdorff_3",
        "ECE_1",
        "ECE_2",
        "ECE_3",
        "ACE_1",
        "ACE_2",
        "ACE_3",
        "AUROC_panc",
        "AUROC_kidn",
        "AUROC_livr",
        "AURC_panc",
        "AURC_kidn",
        "AURC_livr",
        "EAURC_panc",
        "EAURC_kidn",
        "EAURC_livr",
        "CRPS_panc",
        "CRPS_kidn",
        "CRPS_livr",
        "NCC_GT1-2",
        "NCC_GT1-3",
        "NCC_GT2-3",
        "NCC_mean"
    ])

#Computing the metrics

for f in l_patients : 
    current_line = pd.DataFrame([apply_metrics(f)])
    df = pd.concat([df,current_line], ignore_index=True)

#Export data
df.to_csv("metrics.csv", index=False)
print(df)