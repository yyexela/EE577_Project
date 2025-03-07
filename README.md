# EE 577 Final Project

This repository contains source code for the EE577 project. The project report will be uploaded here as well when completed.

## Getting Started

To set up the environment, I use the `conda` python package manager, all packages are listed in `environment.yml`. To create the environment, run:
```bash
conda env create -f environment.yml
```

To activate the environment, run:
```bash

conda activate ee577
```

## Dataset

The `UPENN_GBM` dataset can be found [here](https://brain.labsolver.org/upenn_gbm.html).

Download the dataset files into a directory on your machine with the following path: `EE577_Project/Datasets/<UPENN_GBM>/`. Then, copy all the `.csv` files into a new directory called `csvs` located at `EE577_Project/Datasets/<UPENN_GBM>/csvs/`.

You can change the expected directory structures at `EE577_Project/src/global_config.py`.

## Directory Structure

```text
EE577_Project
├── README.md
├── Datasets
│   └── UPENN_GBM
├── environment.yml
├── Images
├── notebooks
│   ├── cnn_autoencoder.ipynb
│   ├── cnn_autoencoder_UPENN.ipynb
│   ├── Radiometric_results_analysis.ipynb
│   ├── Radiomic_feature_analysis_v2.py
│   └── UPENN_GBM_parsing.ipynb
├── results
│   ├── model_results_summary_for_Linear_Model.csv
│   └── model_results_summary_for_Random_Forest.csv
├── scripts
│   ├── cnn_autoencoder.py
│   ├── cnn_autoencoder_UPENN.py
│   ├── prepare_dataset.py
│   ├── Radiomic_feature_analysis_v2.py
│   ├── transformer_pretrained.py
│   └── transformer_scratch.py
└── src
```

`README.md`:
- This file

`Datasets`:
- Contains all dataset related files.

`environment.yml`:
- Conda environment packages and their versions.

`Images`:
- Contains media (such as results from training).

`results`:
- TODO: Heather

`notebooks`:
- Contains helpful notebooks.
  - `cnn_autoencoder_CIFAR10.ipynb`:
    - This notebook contains the code to train a CNN Autoencoder on CIFAR10.
  - `cnn_autoencoder_UPENN.ipynb`: 
    - This notebook contains code to train a CNN Autoencoder on UPENN_GBM.
  - `UPENN_GBM_parsing.ipynb`: 
    - Helpful notebook parsing the UPENN_GBM dataset to get an idea of what's in it. (TODO: Heather, anything else interesting about this notebook?)
  - `Radiometric_results_analysis.ipynb`: 
    - TODO: Heather
  - `Radiomic_feature_analysis_v2.py`: 
    - TODO: Heather

`scripts`:
- Contains helpful scripts.
  - `cnn_autoencoder_CIFAR10.py`: 
    - The script version of `cnn_autoencoder_CIFAR10.ipynb`.
  - `cnn_autoencoder_UPENN.py`: 
    - The script version of `cnn_autoencoder_UPENN.ipynb`.
  - `prepare_dataset.py`: 
    - This script creates the directory `EE577_Project/Datasets/UPENN_GBM/numpy_conversion_struct_channels/`, which contains the preprocessed images from UPENN_GBM. This file has to be run first before training any models.
    - You can create different pre-processed datasets, and you can change which ones are being used to train models by modifying `upenn_out_dir` in `EE577_Project/src/global_config.py`.
  - `transformer_pretrained.py`: 
    - Trains an MLP head on the embeddings of the UPENN_GBM dataset. The embeddings are obtained from a pre-trained DINOv2 model from HuggingFace.
  - `transformer_scratch.py`: 
    - Trains a Vision Transformer encoder with an MLP head on the UPENN_GBM.
  - `Radiomic_feature_analysis_v2.py`: 
    - TODO: Heather

`src`:
- Contains the source code for the notebooks and scripts. In particular, model definitions and training loops are here, as well as any useful helper functions.

## Contributors

This repository was created and developed by Alexey Yermakov and Heather Wood.

## MISC

This repository contains code from the following sources:

- [DigitalOcean CNN AutoEncoder](https://www.digitalocean.com/community/tutorials/convolutional-autoencoder)
- [UPENN-GBM Dataset Parsing GitHub](https://github.com/LabAIRT/SpotTune_MGMT_prediction)
- [UPENN-GBM Dataset Parsing Paper](https://iopscience.iop.org/article/10.1088/2057-1976/ad6573)