# Disclaimer: code copied from https://github.com/LabAIRT/SpotTune_MGMT_prediction

###############################
# Imports # Imports # Imports #
###############################

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from PIL import Image
from typing import Any, Literal
import torchvision.datasets as tvd
from collections import OrderedDict
import torchvision.transforms.v2 as v2
import src.global_config as global_config
from sklearn.model_selection import train_test_split

# Load config
config = global_config.config

#####################################
# Functions # Functions # Functions #
#####################################

def retrieve_data(modality='T1'):
    # feature csv locations, genomic info is stored in the clinical info csv
    clinical_info = pd.read_csv(os.path.join(config.upenn_csv_dir, 'UPENN-GBM_clinical_info_v2.1.csv'))
    
    # maybe useful in the future, pulls all modalities and stores them into a dictionary of DataFrames 
    features_csvs = [os.path.join(config.upenn_csv_dir, f) for f in os.listdir(config.upenn_csv_dir) if f.endswith('.csv')]
    features_dfs = OrderedDict({os.path.split(f)[-1].strip('.csv'): pd.read_csv(f) for f in features_csvs})
    
    # Set the index to the Patient ID for ease of use and comparison
    clinical_info.set_index('ID', inplace=True)
    for f in features_dfs:
        features_dfs[f].set_index('SubjectID', inplace=True)
    
            # quick (and dirty) way of pulling specific modality dfs and storing them separately
    search_key_ed = '_automaticsegm_'+modality+'_ED'
    search_key_et = '_automaticsegm_'+modality+'_ET'
    search_key_nc = '_automaticsegm_'+modality+'_NC'
    t1_segm_ed = [df for key, df in features_dfs.items() if search_key_ed in key][0]
    t1_segm_et = [df for key, df in features_dfs.items() if search_key_et in key][0]
    t1_segm_nc = [df for key, df in features_dfs.items() if search_key_nc in key][0]
    
    ##################################################################################
    ######## rearrange some features that should be sequential #######################
    ##################################################################################
    match_str = 'Histogram_Bins-16_Bins-16_Bin-'
    match_hist_str = 'Histogram_Bins-16_Bins-16_'
    
    # retrieve list of column names that need to be sequential
    columns_to_sort_ed = [col for col in t1_segm_ed.columns if match_str in col]
    columns_to_sort_et = [col for col in t1_segm_et.columns if match_str in col]
    columns_to_sort_nc = [col for col in t1_segm_nc.columns if match_str in col]
    
    # sort those column names based on 2 criteria
    #   - Freq vs Prob
    #   - Bin number
    col_sort_ed = sorted(columns_to_sort_ed, key=lambda x: (x.split('_')[-1], int(x.split('_')[-2].split('-')[-1])))
    col_sort_et = sorted(columns_to_sort_et, key=lambda x: (x.split('_')[-1], int(x.split('_')[-2].split('-')[-1])))
    col_sort_nc = sorted(columns_to_sort_nc, key=lambda x: (x.split('_')[-1], int(x.split('_')[-2].split('-')[-1])))
    
    # separate out the columns into 3 groups: histogram bin values, other histogram features, everything else
    t1_segm_ed_diff = t1_segm_ed[t1_segm_ed.columns.difference(col_sort_ed)]
    col_rem_hist_ed = [col for col in t1_segm_ed_diff.columns if match_hist_str in col]
    t1_segm_ed_diff_part2 = t1_segm_ed_diff[t1_segm_ed_diff.columns.difference(col_rem_hist_ed)]
    t1_segm_ed_remhist = t1_segm_ed_diff[col_rem_hist_ed]
    t1_segm_ed_sorted = t1_segm_ed[col_sort_ed]
    # join the 3 groups back together, this time in the desired sequential order
    t1_segm_ed = t1_segm_ed_diff_part2.join(t1_segm_ed_sorted, how='inner').join(t1_segm_ed_remhist, how='inner')
    
    # repeat for other tumor segments
    t1_segm_et_diff = t1_segm_et[t1_segm_et.columns.difference(col_sort_et)]
    col_rem_hist_et = [col for col in t1_segm_et_diff.columns if match_hist_str in col]
    t1_segm_et_diff_part2 = t1_segm_et_diff[t1_segm_et_diff.columns.difference(col_rem_hist_et)]
    t1_segm_et_remhist = t1_segm_et_diff[col_rem_hist_et]
    t1_segm_et_sorted = t1_segm_et[col_sort_et]
    t1_segm_et = t1_segm_et_diff_part2.join(t1_segm_et_sorted, how='inner').join(t1_segm_et_remhist, how='inner')
    
    t1_segm_nc_diff = t1_segm_nc[t1_segm_nc.columns.difference(col_sort_nc)]
    col_rem_hist_nc = [col for col in t1_segm_nc_diff.columns if match_hist_str in col]
    t1_segm_nc_diff_part2 = t1_segm_nc_diff[t1_segm_nc_diff.columns.difference(col_rem_hist_nc)]
    t1_segm_nc_remhist = t1_segm_nc_diff[col_rem_hist_nc]
    t1_segm_nc_sorted = t1_segm_nc[col_sort_nc]
    t1_segm_nc = t1_segm_nc_diff_part2.join(t1_segm_nc_sorted, how='inner').join(t1_segm_nc_remhist, how='inner')
    #################################################################################
    #################################################################################
        
    # pulling the manually refined segmentation data
    search_key_ed = '_segm_'+modality+'_ED'
    search_key_et = '_segm_'+modality+'_ET'
    search_key_nc = '_segm_'+modality+'_NC'
    t1_man_segm_ed = [df for key, df in features_dfs.items() if search_key_ed in key][0]
    t1_man_segm_et = [df for key, df in features_dfs.items() if search_key_et in key][0]
    t1_man_segm_nc = [df for key, df in features_dfs.items() if search_key_nc in key][0]
    
    ##################################################################################
    ######## rearrange some features that should be sequential for manually segmented data #######################
    ##################################################################################
    t1_man_segm_ed_diff = t1_man_segm_ed[t1_man_segm_ed.columns.difference(col_sort_ed)]
    t1_man_segm_ed_diff_part2 = t1_man_segm_ed_diff[t1_man_segm_ed_diff.columns.difference(col_rem_hist_ed)]
    t1_man_segm_ed_remhist = t1_man_segm_ed_diff[col_rem_hist_ed]
    t1_man_segm_ed_sorted = t1_man_segm_ed[col_sort_ed]
    t1_man_segm_ed = t1_man_segm_ed_diff_part2.join(t1_man_segm_ed_sorted, how='inner').join(t1_man_segm_ed_remhist, how='inner')
    
    t1_man_segm_et_diff = t1_man_segm_et[t1_man_segm_et.columns.difference(col_sort_et)]
    t1_man_segm_et_diff_part2 = t1_man_segm_et_diff[t1_man_segm_et_diff.columns.difference(col_rem_hist_et)]
    t1_man_segm_et_remhist = t1_man_segm_et_diff[col_rem_hist_et]
    t1_man_segm_et_sorted = t1_man_segm_et[col_sort_et]
    t1_man_segm_et = t1_man_segm_et_diff_part2.join(t1_man_segm_et_sorted, how='inner').join(t1_man_segm_et_remhist, how='inner')
    
    t1_man_segm_nc_diff = t1_man_segm_nc[t1_man_segm_nc.columns.difference(col_sort_nc)]
    t1_man_segm_nc_diff_part2 = t1_man_segm_nc_diff[t1_man_segm_nc_diff.columns.difference(col_rem_hist_nc)]
    t1_man_segm_nc_remhist = t1_man_segm_nc_diff[col_rem_hist_nc]
    t1_man_segm_nc_sorted = t1_man_segm_nc[col_sort_nc]
    t1_man_segm_nc = t1_man_segm_nc_diff_part2.join(t1_man_segm_nc_sorted, how='inner').join(t1_man_segm_nc_remhist, how='inner')
    #################################################################################
    #################################################################################
        
    # pull out the MGMT labels from clinical info, convert to dummy values (0s and 1s) for classification, drop Not Available
    genomics = clinical_info['MGMT']
    truthy_dummies = pd.get_dummies(genomics)
    mgmt_class = truthy_dummies[['Methylated', 'Unmethylated']][np.logical_and(truthy_dummies['Not Available'] != 1, truthy_dummies['Indeterminate'] != 1)]
    
    # match and join classifiers with features for the complete dataframe 
    t1_mgmt_df = mgmt_class.join(t1_segm_ed, how='inner').join(t1_segm_et, how='inner').join(t1_segm_nc, how='inner')
    t1_man_mgmt_df = mgmt_class.join(t1_man_segm_ed, how='inner').join(t1_man_segm_et, how='inner').join(t1_man_segm_nc, how='inner')
    
    
    man_index = t1_man_mgmt_df.index.values
    # make a deep copy so as not to affect the original df
    t1_comb_mgmt_df = t1_mgmt_df.copy(deep=True)
    
    # need to convert the values to be changed to NaNs (essentially empty the values) so that combine_first can be used
    for ind in man_index:
        t1_comb_mgmt_df.loc[ind] = np.nan
        
    # fills all values of NaN in the first df with the values from the second
    t1_comb_mgmt_df = t1_comb_mgmt_df.combine_first(t1_man_mgmt_df)
    
    t1_mgmt_df[t1_mgmt_df.isnull().any(axis=1)].index.tolist()
    
    # dropping rows with nan values
    to_drop = t1_mgmt_df[t1_mgmt_df.isnull().any(axis=1)].index.tolist()
    t1_mgmt_df.drop(labels=to_drop,
                    axis=0,
                    inplace=True)

    # feature selection, dropping unneeded features
    feature_to_remove = ['_OrientedBoundingBoxSize',
                         '_PerimeterOnBorder',
                         '_PixelsOnBorder',
                         'Bins-16_Maximum',
                         'Bins-16_Minimum',
                         'Bins-16_Range']

    feature_mask = t1_mgmt_df.columns.str.contains('|'.join(feature_to_remove))
    feature_column_names = t1_mgmt_df.loc[:, feature_mask].columns.tolist()
    # Don't use difference, it shuffles the columns
    t1_mgmt_df.drop(feature_column_names, axis=1, inplace=True)
    t1_man_mgmt_df.drop(feature_column_names, axis=1, inplace=True)
    t1_comb_mgmt_df.drop(feature_column_names, axis=1, inplace=True)

    return t1_mgmt_df, t1_man_mgmt_df, t1_comb_mgmt_df

def split_image_v2(patients, seed=42):
    """
    splits and scales input dataframe# and outputs as ndarray, assumes binary categories in the first two columns of the dataframe
    """
    X = patients.index
    y = patients
    
    # Separate into train and test datasets.
    # train_test_split automatically shuffles and splits the data following predefined sizes can revisit if shuffling is not a good idea
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    
    return  X_train, y_train, X_test, y_test

def retrieve_patients():
    """
    Obtains labels for dataset given list of modalities and type of classifier

    Alexey Note:
    - Use survival as classifier, creates bins for patients and removes patients without survival data
    - Returns one-hot encoded list of patients with survival data and their respective bin
    """
    csv_dir = config.upenn_csv_dir
    image_dir_ = config.upenn_image_dir
    classifier = config.classifier

    # feature csv locations, genomic info is stored in the clinical info csv
    clinical_info = pd.read_csv(os.path.join(csv_dir, '../UPENN-GBM_clinical_info_v2.1.csv'))
    clinical_info.set_index('ID', inplace=True)
    class_of_interest = clinical_info[classifier]

    struct_mods = ['T1', 'T1GD', 'T2', 'FLAIR']
    modality = struct_mods

    if 'Survival' in classifier:
        survival = class_of_interest.drop(class_of_interest[class_of_interest == 'Not Available'].index)
        survival = pd.DataFrame(survival, dtype=int)
        survival['survival_bin'] = pd.cut(x=survival[classifier], 
                                          bins = [0, 100, 200, 300, 400, 500, 700, np.inf])
                                          #bins = [0, 365, 730, 1095, 1460, 1825, np.inf])
        final_class = pd.get_dummies(survival['survival_bin'])
    else:
        raise Exception("Not implemented")

    mod_patients = {}
    if np.any([True if mod in struct_mods else False for mod in modality]):
        scan = os.scandir(os.path.join(image_dir_, 'images_structural'))
        mod_patients['structural'] = [d.name for d in scan if d.name in final_class.index.tolist() and ('_21' not in d.name)]
    
    patients = pd.DataFrame(final_class)
    for mod in mod_patients:
        patients = patients.loc[[pat for pat in mod_patients[mod] if pat in patients.index]]
    idx_to_remove = [label for label in patients.index.tolist() if '_21' in label]
    patients = patients.drop(idx_to_remove)

    return patients