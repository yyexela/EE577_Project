import os
import sys
import numpy as np

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import layers, models,regularizers,losses
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor

from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


pkg_path = str(Path(os.path.abspath('')).parent.absolute())
sys.path.insert(0, pkg_path)
print(pkg_path)

# Choose model type:
# model_label='Random_Forest'
model_label='Linear_Model'
results_file = f'project/results/model_results_summary_for_{model_label}.csv'
results_file_path= os.path.join(pkg_path,results_file)
print(results_file_path)
# reference path: /home/ee577/project/results

# data_path='/home/ee577/project/Datasets/UPENN_GBM/'
data_path=pkg_path+'/Datasets/UPENN_GBM/' # data_availability
csv_path=data_path+'csvs/'


data_availability = pd.read_csv(os.path.join(data_path, 'UPENN-GBM_data_availability.csv'))
data_availability = data_availability[data_availability['Overall Survival'] != 'not available']
patient_ids = data_availability['ID']

clinical_info = pd.read_csv(os.path.join(data_path, 'UPENN-GBM_clinical_info_v2.1.csv'))
clinical_info = clinical_info[clinical_info['ID'].isin(patient_ids)]
clinical_info = clinical_info.set_index('ID')
survival_days = clinical_info['Survival_from_surgery_days_UPDATED']

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.0001, verbose=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True,verbose=0)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True,verbose=0)

def drop_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    return df

def train_transform(radiomeric_features, survival_days, seed=42):
    X = radiomeric_features
    Y = survival_days
    X, Y = X.align(Y, join='inner', axis=0)
    Y= Y.astype(int) 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def get_linear_model(X_train_shape, drop_rate = 0.3, regularizer=tf.keras.regularizers.L1L2(
    l1=0.00001, l2=0.001
), delta=5):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(X_train_shape[1],))) 
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizer))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizer))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizer))
    model.add(layers.Dense(1))  
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(delta=delta), metrics=['mae'])
    return model #'mean_squared_error'

def get_RF_regressor(n_estimators=40, max_depth=10, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    return model

def train_RF_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def train_linear_model(model, X_train, y_train, epochs=50, batch_size=32):
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    history = model.fit(
    X_train, y_train,  
    epochs=epochs,                 
    batch_size=batch_size,             
    validation_split=0.2,      
    callbacks=[early_stopping, model_checkpoint, lr_reduction]  
    )
    return history 
   
def adjusted_r2(r2, n, p):
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2

def process_fold(X_train, X_test, y_train, y_test, results, epochs, batch_size, seed):
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Get the model
    model = get_linear_model(X_train.shape)  # Adjusted for the shape

    # Train the model
    train_linear_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    # Calculate R2
    r2 = r2_score(y_test, y_pred)
    n = X_train.shape[0]  # Number of samples
    p = X_train.shape[1]  # Number of features
    adj_r2 = adjusted_r2(r2, n, p)  # Adjusted R2

    # Store results
    results['mse'].append(mse)
    results['adj_r2'].append(adj_r2)

def cross_validate_model(radiomic_features, survival_days, seed=42, k_folds=1, epochs=40, batch_size=32):
    X = radiomic_features
    Y = survival_days
    X, Y = X.align(Y, join='inner', axis=0)
    Y = Y.astype(int)

    # If k_folds = 1, use a simple train-test split
    if k_folds == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
        kf = None  # No cross-validation
    else:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    results = {'adj_r2': [], 'mse': []}

    if kf:  # If cross-validation is enabled
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
            process_fold(X_train, X_test, y_train, y_test, results, epochs, batch_size, seed)

    else:  # If no cross-validation, just a train-test split
        process_fold(X_train, X_test, y_train, y_test, results, epochs, batch_size, seed)

    # Calculate the average performance metrics
    avg_mse = np.mean(results['mse'])
    avg_adj_r2 = np.mean(results['adj_r2'])
    return avg_mse, avg_adj_r2

# Example of Adjusted R² function
def adjusted_r2(r2, n, p):
    """Calculate adjusted R²."""
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2



def run_experiment_for_file(file_path, random_seeds=[1, 2, 3, 4, 5], Random_Forest=False):
    df = pd.read_csv(os.path.join(csv_path, file_path))
    df = df[df['SubjectID'].isin(patient_ids)]
    df = df.set_index('SubjectID')
    df = drop_correlated_features(df)
    df = df.dropna()
    results = {'mse': [], 'r2': []}
    best_r2 = -float('inf')

    for seed in random_seeds:
        # Split data and train model
        tf.random.set_seed(seed)
        X_train, X_test, y_train, y_test = train_transform(df, survival_days, seed)
        if Random_Forest:
            model = get_RF_regressor(random_state=seed)
            history = train_RF_model(model, X_train, y_train)
        else:
            model = get_linear_model(X_train.shape)
            history = train_linear_model(model, X_train, y_train)

        # Makes predictions
        y_pred = model.predict(X_test)
        
        # Calculates MSE and R2 score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Stores results
        results['mse'].append(mse)
        results['r2'].append(r2)
        if r2 > best_r2:
            best_r2 = r2
        else:
            pass

    # Calculates average and std dev for MSE and R2
    mse_avg = np.mean(results['mse'])
    mse_std = np.std(results['mse'])
    r2_avg = np.mean(results['r2'])
    r2_std = np.std(results['r2'])

    # Save the best model to a file
   
    
    return {
        'file_name': file_path,
        'mse_avg': mse_avg,
        'mse_std': mse_std,
        'r2_avg': r2_avg,
        'r2_std': r2_std,
        'mse': results['mse'],
        'r2': results['r2'],
        'best_r2': best_r2,
    }


# Loops through all files starting with 'Radiomic_Features_CaPTk'
# file_names = [f for f in os.listdir(csv_path) if f.startswith('Radiomic_Features_CaPTk')]

# Results
all_results = []

# Load existing results if the file exists
""" if os.path.exists(results_file_path):
    Save_point_df = pd.read_csv(results_file_path)
    all_results = Save_point_df.to_dict(orient='records')  # Convert to list of dicts for appending
    print("Existing results found and loaded.")

for file_name in file_names:
    # Check if the result for the current file already exists
    existing_result = any(file_name in str(result) for result in all_results)
    
    if not existing_result:  # Only process if result doesn't exist
        if model_label=="Random_Forest":
            result=run_experiment_for_file(file_name, Random_Forest=True)
        else:
            result = run_experiment_for_file(file_name)
        all_results.append(result)
        
        # Save progress after each iteration
        Save_point_df = pd.DataFrame(all_results)
        Save_point_df.to_csv(results_file_path, index=False)
        print(f"Completed processing for {file_name}")
    else:
        print(f"Skipping {file_name}, results already exist.")"""

# Final summary output
summary_df = pd.DataFrame(all_results)
print(summary_df)


# summary_df.to_csv('model_results_summary.csv', index=False)