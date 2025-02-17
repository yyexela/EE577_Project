Environment:
- `conda activate ee577`

Notes:
- Stratified sampling of data
- 3D volumetric images
- Not all patients have lifespan data
- Lifespan is categorized in ranges

Considerations:
- Use masked or unmasked data
- How many "slices" to use
- Use 3D volumetric data or 2D slice (centered)
- Data preprocessing (subtract mean/stdev)
- Data augmentation
- Not enough data in a single modality for a CNN

Code pulled from:
- [CNN AutoEncoder Source](https://www.digitalocean.com/community/tutorials/convolutional-autoencoder)
- [UPENN-GBM parsing source](https://github.com/LabAIRT/SpotTune_MGMT_prediction)