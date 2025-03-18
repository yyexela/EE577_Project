Environment:
- `conda activate ee577`

Important Files:

### Data generating files ###
File: Radiomic_feature_analysis_v2.py
In: csvs containing features obtained using CaPTk package with ET, ED, and NC segmentation masks
Out: functional_results/model_results_summary_for_Linear_Model.csv
Purpose: Establish which modalities may be most predictive of survival

File: notebooks/Shap_values.ipynb
In: Top models from Radiomic_feature_analysis_v2.py
Out: functional_results
/shap_DSC_ap_rCBV_ED.csv
/shap_DTI_AD_NC_.csv
/selected_features_DSC.csv
/selected_features_DTI.csv
Purpose: Feature selection for training first U-net

File: notebooks/data_prep_for_segModels.ipynb
In: 3D tumor data, feature csvs, survival/clinical data
Out: pickel files containing masked, scaled and cropped tumors (X) paired with either another modality (X1, X2) or with features (usually y except during test train split) and scaled/normalized Z (survival data), also tumor box data
Purpose: Prepare and pair up different data sources for U-net and U-net derived models

File: scripts/Transfer_learning_feat_trainer.py
In: Prepared 3D tumor segmentations paired with feature data, shapley values for the weighted MSE cost function
Out: best_models_unet
/DSC_feat_best_model.pth
/DTI_feat_best_model.pth
/functional_results
/DSC_feat_loss.csv
functional_results
/DTI_feat_loss.csv
Purpose: Train 3D U-net models to predict features based on segmentation mask and shapley values

File: scripts/Transfer_learning_surv_trainer.py
In: Prepared 3D tumor segmentations paired with normalized survival data
Out: best_models_unet
/DSC_surv_best_model.pth
/DTI_surv_best_model.pth
/functional_results
/DSC_surv_loss.csv
functional_results
/DTI_surv_loss.csv
Purpose: Train 3D U-net models to predict survival data

File: notebooks /Salience_maps.ipynb
In: best models, paired data, 
Out: Saliency maps, rescaled survival predictions, other model metrics
Purpose: To provide data for U-Net model analysis

### Data analysis files ###
File: notebooks/feature_analysis.ipynb
In: results stored in functional_results
Out: Graphs of the simple 3 layered FNN's r-squared values, Top 10 Summed Absolute Shapley Values for each modality, scaled and unscaled survival histograms, first attempt at saliency maps (note tumor locations are not yet paired up here)
Purpose: To analyze the simplier models and data that the U-nets will be trained on

File: notebooks/UNet_results_analysis.ipynb
In: results stored in functional_results
Out: Graphs of the training and validation loss
Purpose: To analyze the results of U-net training
Code pulled from:
- [CNN AutoEncoder Source](https://www.digitalocean.com/community/tutorials/convolutional-autoencoder)
- [UPENN-GBM parsing Source](https://github.com/LabAIRT/SpotTune_MGMT_prediction)
- [UPENN-GBM parsing Paper](https://iopscience.iop.org/article/10.1088/2057-1976/ad6573)
