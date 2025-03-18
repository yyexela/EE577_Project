# %%

import os
from pathlib import Path
import sys
import numpy as np
import pickle
import pandas as pd

pkg_path = str(Path(os.path.abspath('')).parent.absolute())
sys.path.insert(0, pkg_path)


data_path=pkg_path+'/results/'

from src import *

# use this if having problems:
# export PYTHONPATH=/home/ee577/project/src:$PYTHONPATH




# %%
pkg_path = str(Path(os.getcwd()).parent.absolute())  # Get parent directory of the current working directory
sys.path.insert(0, pkg_path)

# If you need to include a 'src' directory relative to the current notebook:
src_path = os.path.abspath(os.path.join(os.getcwd(), '../src'))  # Adjust path relative to the current working directory
sys.path.insert(0, src_path)

print(f"Path to the 'src' directory: {src_path}")

# Now you can import your modules from the 'src' directory
from src import *

# Load config file
config = global_config.config
config.device = 'cuda:1'
torch.manual_seed(config.seed)
print(f"This is the output directory: {config.dataset_dir}")

# %%
preprocess_config = {
    'modality': ['DTI'], #['T2', 'FLAIR', 'T1', 'T1GD']
    'image_type': 'autosegm',
    'window': (140, 172, 164),
    'pad_window': (70, 86, 86),
    'crop': True,
    'window_idx': ((0, 144), (31, 214), (44,209)),
    'down_factor': 0.5,
    'augments': ['base'] #'base', 'flip', 'rotate', 'noise', 'deform'
}

paths, image_dict,tumor_boxes=data_prep.convert_image_data_mod(**preprocess_config)

raise Exception("intentional")


# %% [markdown]
# output_dir = '/home/ee577/project/Datasets/UPENN_GBM/DTI_numpy_files'
# 
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# 
# # Iterate over each patient and their modalities in image_container
# for pat, mod_arr in image_dict.items():
#     # Create a directory for each patient
#     patient_dir = os.path.join(output_dir, pat)
#     if not os.path.exists(patient_dir):
#         os.makedirs(patient_dir)
# 
#     # Check if 'AD' modality exists in the current patient's mod_arr
#     if 'AD' in mod_arr:
#         img_data = mod_arr['AD']  # Get the image data for the 'AD' modality
#         
#         # Construct the filename for the 'AD' modality (you can adjust the naming convention)
#         filename = f"{pat}_AD.npy"
#         filepath = os.path.join(patient_dir, filename)
#         
#         # Save the image data as a .npy file
#         np.save(filepath, img_data)  # Save the image as a .npy file
# 
#         print(f"Saved image for patient {pat}, modality AD at {filepath}")
#     else:
#         print(f"No 'AD' modality found for patient {pat}. Skipping.")
# 
# 
# 
# 

# %%
# getting clinical data
clinical_info = pd.read_csv(os.path.join(config.upenn_dir, 'UPENN-GBM_clinical_info_v2.1.csv'))
#display(clinical_info)
print(clinical_info.columns)
print(list(clinical_info.dtypes))

pd.set_option("display.max_rows", 40)

survival_days = clinical_info['Survival_from_surgery_days_UPDATED']
survival_days = survival_days.drop(survival_days[survival_days == 'Not Available'].index)

print("statistics:")
display(survival_days.describe())

clinical_info.set_index('ID', inplace=True)

clinical_factors = clinical_info[['IDH1', 'MGMT']]

#list(features['Selected_features'].values)
features=pd.read_csv(data_path+'selected_features_DTI.csv')
all_features_DTI_AD_NC=pd.read_csv(os.path.join(config.upenn_dir,"csvs", 'Radiomic_Features_CaPTk_automaticsegm_DTI_AD_NC.csv'))

selected_features = list(features['Selected_features'].values)
matching_columns = [col for col in all_features_DTI_AD_NC.columns for feature in selected_features if feature in col]
filtered_data = all_features_DTI_AD_NC[matching_columns]
filtered_data_features = all_features_DTI_AD_NC[['SubjectID'] + matching_columns]
print(filtered_data_features.head())

filtered_data_features = filtered_data_features.set_index('SubjectID')
numerical_columns = filtered_data_features.select_dtypes(include=['number']).columns
if len(numerical_columns) == len(filtered_data_features.columns):
    print("All features are numerical.")
else:
    print("Some features are not numerical.")

# %%
idh1_encoded = clinical_info['IDH1'].map({'Wildtype': 0, 'NOS/NEC': 1})
mgmt_encoded = clinical_info['MGMT'].map({'Methylated': 1, 'Unmethylated': 0})

idh1_encoded.fillna(idh1_encoded.mean(), inplace=True)
mgmt_encoded.fillna(mgmt_encoded.mean(), inplace=True)
encoded_df = pd.DataFrame({
    'IDH1_encoded': idh1_encoded,
    'MGMT_encoded': mgmt_encoded
})

merged_data = filtered_data_features.merge(encoded_df, left_index=True, right_index=True, how='left')

print(merged_data.head())

# %%
def save_image_feature_pairing(image_containter, feature_df, output_file):
    """
    Function to load image data for a specific modality ('AD') and pair it with the corresponding feature data 
    for each patient, then save the paired data as a pickle file.

    Args:
        image_containter (OrderedDict): A dictionary of images for each patient and modality.
        feature_df (pd.DataFrame): A DataFrame containing the features for each patient.
        output_file (str): Path to the output pickle file.
    """
    # Initialize lists to hold image data and corresponding feature data
    X = []  # Image data
    y = []  # Feature data
    
    # Loop over each patient in the feature dataframe
    for patient_id, row in feature_df.iterrows():
        # Check if the patient has an 'AD' modality image available
        if patient_id in image_containter:
            # Retrieve the image data for 'AD' modality
            image_data = image_containter[patient_id].get('AD')
            if image_data is not None:
                # Add the image data (X) and the corresponding feature data (y)
                X.append(image_data)
                y.append(row.values)  # Assuming row contains the feature values
    
    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Save the image-feature pairs as a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump((X, y), f)
    
    print(f"Saved paired data to {output_file}")




# %%
out_dir = '/home/ee577/project/results/DTI_AD_NC_paired_data.pkl' 

save_image_feature_pairing(image_dict, merged_data, out_dir)



