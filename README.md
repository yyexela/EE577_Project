Environment:
- `conda activate ee577`

Notes:
- Stratified sampling of data
- 3D volumetric images
- Not all patients have lifespan data
- Lifespan is categorized in ranges

```text
Tumor box data:
            X min       X max       Y min       Y max       Z min       Z max
count  585.000000  585.000000  585.000000  585.000000  585.000000  585.000000
mean    49.071795  112.611966   95.182906  173.357265   93.003419  151.056410
std     17.840378   19.807244   26.582562   23.983311   30.971651   29.188368
min      0.000000   49.000000   31.000000   86.000000   44.000000   82.000000
25%     35.000000   99.000000   80.000000  156.000000   63.000000  121.000000
50%     48.000000  115.000000   92.000000  176.000000   92.000000  160.000000
75%     61.000000  130.000000  115.000000  193.000000  121.000000  179.000000
max    106.000000  144.000000  174.000000  214.000000  159.000000  209.000000
```

Considerations:
- Use masked or unmasked data
- How many "slices" to use
- Use 3D volumetric data or 2D slice (centered)
- Data preprocessing (subtract mean/stdev)
- Data augmentation
- Not enough data in a single modality for a CNN

Code pulled from:
- [CNN AutoEncoder Source](https://www.digitalocean.com/community/tutorials/convolutional-autoencoder)
- [UPENN-GBM parsing Source](https://github.com/LabAIRT/SpotTune_MGMT_prediction)
- [UPENN-GBM parsing Paper](https://iopscience.iop.org/article/10.1088/2057-1976/ad6573)
