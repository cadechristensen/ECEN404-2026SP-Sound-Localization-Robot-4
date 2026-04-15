import os
import re
import glob
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# CONFIGURATION
# ==========================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_SCRIPT_DIR, 'models')
DATA_DIR = 'raw'
MODEL_FILENAME = 'distance_model_rawTry2.joblib'
FEATURE_NAMES_FILENAME = 'feature_names_rawTry2.joblib'

# Audio Processing Parameters
HOP_LENGTH = 512
FRAME_LENGTH = 2048
SAMPLE_RATE = 48000

# --- Toggles ---
SPLIT_FOR_VALIDATION = False 
AUTO_RESERVE_TEST_SET = False 
USE_LOG_DISTANCE = True

# ==========================================
# STEP 1: EXPANDED FEATURE EXTRACTION
# ==========================================
def extract_features(y, sr):
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    
    if len(y_trimmed) < FRAME_LENGTH:
        y_trimmed = y
        
    features = {}
        
    # 1. RMS (Volume)
    rms = librosa.feature.rms(y=y_trimmed, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max) 
    features['rms_mean'] = np.mean(rms_db)
    features['rms_std'] = np.std(rms_db)
    
    # 2. Spectral Centroid (Brightness)
    spec_cent = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    features['spec_cent_mean'] = np.mean(spec_cent)
    features['spec_cent_std'] = np.std(spec_cent)
    
    # NEW 3. Spectral Bandwidth (Frequency Spread)
    spec_bw = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    features['spec_bw_mean'] = np.mean(spec_bw)
    features['spec_bw_std'] = np.std(spec_bw)
    
    # NEW 4. Spectral Rolloff (High-frequency decay)
    rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_std'] = np.std(rolloff)
    
    # NEW 5. Zero-Crossing Rate (Noisiness/Harshness)
    zcr = librosa.feature.zero_crossing_rate(y=y_trimmed, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # 6. MFCCs (Timbre/Texture)
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    for i in range(13):
        features[f'mfcc_mean_{i+1}'] = mfccs_mean[i]
        features[f'mfcc_std_{i+1}'] = mfccs_std[i]
        
    return features

def process_audio_files(data_dir):
    data = []
    wav_files = glob.glob(os.path.join(data_dir, '*.wav'))

    if not wav_files:
        print(f"Error: No .wav files found in directory '{data_dir}'.")
        return pd.DataFrame()

    print(f"Found {len(wav_files)} total .wav files. Extracting 36 acoustic features per file...")
    
    training_file_count = 0
    test_file_count = 0

    for f in wav_files:
        filename = os.path.basename(f)
        
        if AUTO_RESERVE_TEST_SET and '_set4' in filename:
            test_file_count += 1
            continue

        match = re.search(r'(\d+)ft', filename)
        if not match:
            continue

        distance = int(match.group(1))

        try:
            y, sr = librosa.load(f, sr=SAMPLE_RATE)
            features = extract_features(y, sr)
            features['distance'] = distance
            data.append(features)
            training_file_count += 1
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    print("-" * 40)
    print(f"Files extracted for model dataset: {training_file_count}")
    print("-" * 40)
    
    return pd.DataFrame(data)

# ==========================================
# STEP 2: TRAIN & TUNE (MAXIMUM COMPUTE)
# ==========================================
def train_model(df):
    print("\n=======================================================")
    print(" INITIATING MAXIMUM OPTIMIZATION SEQUENCE")
    print(f" LOG DISTANCE TARGETS: {USE_LOG_DISTANCE}")
    print("=======================================================\n")
    
    # 1. SET THE TARGET VARIABLE
    if USE_LOG_DISTANCE:
        y = np.log(df['distance'])  # The Physics Hack
    else:
        y = df['distance']          # Standard Linear Approach
        
    X = df.drop('distance', axis=1)
    
    if SPLIT_FOR_VALIDATION:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=df['distance']
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    print(f"Total available acoustic features: {len(X.columns)}")

    # 2. RECURSIVE FEATURE ELIMINATION (RFECV)
    print("\nPhase 1: Running Recursive Feature Elimination (RFECV)...")
    print("The model is mathematically finding the exact perfect combination of features.")
    
    rf_for_selection = RandomForestRegressor(n_estimators=100, random_state=42)
    rfecv = RFECV(
        estimator=rf_for_selection, 
        step=1, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        min_features_to_select=5,
        n_jobs=-1 # Uses all CPU cores
    )
    rfecv.fit(X_train, y_train)
    
    optimal_num_features = rfecv.n_features_
    top_features = X_train.columns[rfecv.support_].tolist()
    
    print(f"\n--> RFECV Complete! Optimal number of features proved to be: {optimal_num_features}")
    print(f"--> Keeping: {', '.join(top_features)}")
    
    # Filter dataset to ONLY the mathematically perfect features
    X_train = X_train[top_features]
    if X_test is not None:
        X_test = X_test[top_features]
        
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(top_features, os.path.join(MODELS_DIR, FEATURE_NAMES_FILENAME))

    # 3. THE GOD-GRID SEARCH
    print("\nPhase 2: Initiating God-Grid Search (This will take a long time)...")
    
    # Sample Weights for extreme edges
    if USE_LOG_DISTANCE:
        train_distances_ft = np.exp(y_train)
    else:
        train_distances_ft = y_train
    
    
    sample_weights = np.where(train_distances_ft >= 8, 4.0, 1.0)    
    #sample_weights = np.where((train_distances_ft <= 3) | (train_distances_ft >= 9), 3.0, 1.0)

    # ==========================================
    # THE MICRO-GRID (Optimized around previous best)
    # ==========================================
    param_grid = {
        # Check just around the winning 200
        'n_estimators': [150, 200, 250],      
        
        # Check just around the winning 10
        'max_depth': [8, 10, 15],             
        
        # Check just around the winning 4
        'min_samples_split': [3, 4, 5],       
        
        # Won at 1 (can't go lower), so just check 1 and 2
        'min_samples_leaf': [1, 2],           
        
        # Lock these in, since they are mathematically proven to be the best for this data!
        'max_features': ['sqrt'],             
        'bootstrap': [False]                  
    }
    
    rf_base = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_base, 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, # BURN THE CPU
        scoring='neg_mean_absolute_error', 
        verbose=1
    )

    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    best_model = grid_search.best_estimator_
    print("\n=======================================================")
    print(" GRID SEARCH COMPLETE! ")
    print("=======================================================")
    print(f"Absolute Best Parameters Found:\n{grid_search.best_params_}")

    if SPLIT_FOR_VALIDATION:
        print("\n--- Internal Validation Score ---")
        y_pred_raw = best_model.predict(X_test)
        
        # Reverse the log if we used it, otherwise leave it alone
        if USE_LOG_DISTANCE:
            y_pred_ft = np.exp(y_pred_raw)
            y_test_ft = np.exp(y_test)
        else:
            y_pred_ft = y_pred_raw
            y_test_ft = y_test
            
        mae = mean_absolute_error(y_test_ft, y_pred_ft)
        r2 = r2_score(y_test_ft, y_pred_ft)
        
        print(f"Mean Absolute Error (MAE): {mae:.3f} feet")
        print(f"R-squared (R²): {r2:.3f}")

    return best_model, top_features

# ==========================================
# STEP 3: PREDICTION ENGINE
# ==========================================
def predict_distance_from_file(model, feature_names, filepath):
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        features_dict = extract_features(y, sr)
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[feature_names] 
        
        predicted_val = model.predict(features_df)[0]
        
        # Check toggle to decide whether to reverse the log
        if USE_LOG_DISTANCE:
            predicted_dist_ft = np.exp(predicted_val) # Physics Hack Translation
        else:
            predicted_dist_ft = predicted_val         # Standard ML prediction
            
        return predicted_dist_ft
    except Exception as e:
        print(f"Error predicting on file {filepath}: {e}")
        return None
    
# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    main_df = process_audio_files(DATA_DIR)

    if main_df.empty:
        print("Exiting script. No data to train on.")
    else:
        trained_model, feature_names = train_model(main_df)

        model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        joblib.dump(trained_model, model_path)
        print(f"\nModel saved successfully as '{model_path}'")
        features_path = os.path.join(MODELS_DIR, FEATURE_NAMES_FILENAME)
        print(f"Feature list saved successfully as '{features_path}'")

        if AUTO_RESERVE_TEST_SET:
            print("\n--- Testing with all reserved 'set4' files ---")
            set4_files = glob.glob(os.path.join(DATA_DIR, '*_set4*.wav'))
            
            if not set4_files:
                print("No 'set4' files found to test.")
            else:
                results = []
                for test_file in set4_files:
                    filename = os.path.basename(test_file)
                    match = re.search(r'(\d+)ft', filename)
                    
                    if not match:
                        continue
                        
                    actual_dist = int(match.group(1))
                    predicted_dist = predict_distance_from_file(trained_model, feature_names, test_file)
                    
                    if predicted_dist is not None:
                        error = predicted_dist - actual_dist
                        results.append({
                            'File': filename,
                            'Actual (ft)': actual_dist,
                            'Predicted (ft)': predicted_dist,
                            'Error (ft)': error
                        })
                
                if results:
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values(by='Actual (ft)') 
                    print("\n" + results_df.to_string(float_format="%.2f", index=False))
                    
                    final_mae = results_df['Error (ft)'].abs().mean()
                    print("-" * 40)
                    print(f"FINAL MODEL MAE (on set4 data): {final_mae:.3f} feet")
                    print("-" * 40)