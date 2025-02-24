import os
import sys
import numpy as np

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import layers, models,regularizers

from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


pkg_path = str(Path(os.path.abspath('')).parent.absolute())
sys.path.insert(0, pkg_path)
data_path=pkg_path+'/Datasets/UPENN_GBM/csvs/'

data_availability = pd.read_csv(os.path.join(data_path, 'UPENN-GBM_data_availability.csv'))
data_availability = data_availability[data_availability['Overall Survival'] != 'not available']
patient_ids = data_availability['ID']

clinical_info = pd.read_csv(os.path.join(data_path, 'UPENN-GBM_clinical_info_v2.1.csv'))
clinical_info = clinical_info[clinical_info['ID'].isin(patient_ids)]
clinical_info = clinical_info.set_index('ID')
survival_days = clinical_info['Survival_from_surgery_days_UPDATED']

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

def drop_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    return df

def train_transform(radiomeric_features, survival_days):
    X = radiomeric_features
    Y = survival_days
    X, Y = X.align(Y, join='inner', axis=0)
    Y= Y.astype(int) 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def get_linear_model(X_train_shape, drop_rate = 0.2, regularizer=tf.keras.regularizers.l2(0.01)):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(X_train_shape[1],))) 
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizer))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizer))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizer))
    model.add(layers.Dense(1))  
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_the_model(model, X_train, y_train, epochs=100, batch_size=32):
    history = model.fit(
    X_train, y_train,  
    epochs=epochs,                 
    batch_size=batch_size,             
    validation_split=0.2,      
    callbacks=[early_stopping, model_checkpoint, lr_reduction]  
    )
    return history

def run_experiment_for_file(file_path, random_seeds=[1, 2, 3, 4, 5]):
    df = pd.read_csv(os.path.join(data_path, file_path))
    df = df[df['SubjectID'].isin(patient_ids)]
    df = df.set_index('SubjectID')
    df = drop_correlated_features(df)

    results = {'mse': [], 'r2': []}
    
    for seed in random_seeds:
        # Split data and train model
        X_train, X_test, y_train, y_test = train_transform(df, survival_days)
        model = get_linear_model(X_train.shape)

        # Train the model with the current random seed
        tf.random.set_seed(seed)
        history = train_the_model(model, X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate MSE and R2 score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results['mse'].append(mse)
        results['r2'].append(r2)

    # Calculate average and std dev for MSE and R2
    mse_avg = np.mean(results['mse'])
    mse_std = np.std(results['mse'])
    r2_avg = np.mean(results['r2'])
    r2_std = np.std(results['r2'])

    return {
        'file_name': file_path,
        'mse_avg': mse_avg,
        'mse_std': mse_std,
        'r2_avg': r2_avg,
        'r2_std': r2_std,
        'mse': results['mse'],
        'r2': results['r2']
    }

# Loop through all files starting with 'Radiomic_Features_CaPTk'
file_names = [f for f in os.listdir(data_path) if f.startswith('Radiomic_Features_CaPTk')]

# Store all results
all_results = []

for file_name in file_names:
    result = run_experiment_for_file(file_name)
    all_results.append(result)

# Create a DataFrame to summarize results
summary_df = pd.DataFrame(all_results)

# Print the summary
print(summary_df)

# Optionally, save the summary as a CSV
summary_df.to_csv('model_results_summary.csv', index=False)



