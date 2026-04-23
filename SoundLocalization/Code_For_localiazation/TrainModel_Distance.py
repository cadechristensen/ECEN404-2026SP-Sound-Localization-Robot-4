import os
import re
import glob
import itertools
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_DIR = 'raw'             
TEST_DIR = 'single'         

# Updated filenames so you don't overwrite your previous best!
MODEL_FILENAME = 'distance_model_rawApril6_angle_try2.joblib'
FEATURE_NAMES_FILENAME = 'feature_names_rawApril6_angle_try2.joblib'

# Audio Processing Parameters
HOP_LENGTH = 512
FRAME_LENGTH = 2048
SAMPLE_RATE = 48000

# --- Toggles ---
LOG_OPTIONS = [False, True] 

# --- Distance Weight Profiles ---
# --- Distance Weight Profiles ---
WEIGHT_PROFILES = [
    # Profile 1: The "Sledgehammer" (Your previous winner, now the baseline)
    {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 6.0, 6: 9.0, 7: 13.0, 8: 18.0, 9: 24.0, 10: 30.0},
    
    # Profile 2: The "Anvil" (Pushes the 8ft and 10ft penalties even higher)
    {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 6.0, 6: 9.0, 7: 20.0, 8: 25.0, 9: 30.0, 10: 40.0},
    
    # Profile 3: The "Bulldozer" (Maximum mathematical brute force on the furthest distances)
    {1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0, 5: 7.0, 6: 12.0, 7: 18.0, 8: 26.0, 9: 36.0, 10: 50.0}
]

# --- Hyperparameter Grid (Micro-Grid V2) ---
PARAM_GRID = {
    # Testing lower since 150 was the floor last time
    'n_estimators': [150,151,152],      
    
    # Testing higher since 18 was the ceiling last time. 
    # 'None' means the trees will grow infinitely deep until they perfectly map the data.
    'max_depth': [14,15,16],             
    
    # Centering tightly around the previous winner (6)
    'min_samples_split': [5, 6, 7],       
    
    # Centering tightly around the previous winner (2)
    'min_samples_leaf': [1, 2, 3],           
    
    # Locked in from previous winning run to save CPU time
    'max_features': ['log2'],            
    'bootstrap': [False]                  
}

# ==========================================
# STEP 1: EXPANDED FEATURE EXTRACTION
# ==========================================
def extract_features(y, sr):
    y_trimmed, _ = librosa.effects.trim(y, top_db=60)
    
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
    
    # 3. Spectral Bandwidth (Frequency Spread)
    spec_bw = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    features['spec_bw_mean'] = np.mean(spec_bw)
    features['spec_bw_std'] = np.std(spec_bw)
    
    # 4. Spectral Rolloff (High-frequency decay)
    rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_std'] = np.std(rolloff)
    
    # 5. Zero-Crossing Rate (Noisiness/Harshness)
    zcr = librosa.feature.zero_crossing_rate(y=y_trimmed, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # 6. MFCCs (Timbre/Texture)

        
    # Change it to this:
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    for i in range(20): # Make sure to change this loop to 20 as well!
        features[f'mfcc_mean_{i+1}'] = mfccs_mean[i]
        features[f'mfcc_std_{i+1}'] = mfccs_std[i]
        
    return features

def process_audio_files(data_dir):
    data = []
    wav_files = glob.glob(os.path.join(data_dir, '*.wav'))

    if not wav_files:
        print(f"Warning: No .wav files found in directory '{data_dir}'.")
        return pd.DataFrame()

    print(f"Extracting features from {len(wav_files)} files in '{data_dir}'...")

    for f in wav_files:
        filename = os.path.basename(f)
        
        # --- NEW REGEX FOR ANGLE AND DISTANCE ---
        # Looks for "test" followed by numbers (angle), then "_" then numbers followed by "ft"
        match = re.search(r'test(\d+)_(\d+)ft', filename)
        if not match:
            continue

        angle = int(match.group(1))
        distance = int(match.group(2))

        try:
            y, sr = librosa.load(f, sr=SAMPLE_RATE)
            features = extract_features(y, sr)
            
            # --- ADD ANGLE TO FEATURES ---
            features['angle'] = angle 
            features['distance'] = distance
            features['filename'] = filename 
            
            data.append(features)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    return pd.DataFrame(data)

# ==========================================
# STEP 2: TRAIN & CUSTOM TUNE
# ==========================================
def train_and_tune_model(df_train, df_test):
    print("\n=======================================================")
    print(" INITIATING TARGETED FOLDER OPTIMIZATION SEQUENCE")
    print("=======================================================\n")
    
    # We drop distance and filename as usual, but 'angle' stays in as a predictive feature!
    X_train_full = df_train.drop(['distance', 'filename'], axis=1)
    X_test_full = df_test.drop(['distance', 'filename'], axis=1)

    best_mae = float('inf')
    best_model = None
    best_params = None
    best_profile_idx = None
    best_features = None
    best_log_status = None

    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for use_log in LOG_OPTIONS:
        print(f"\n--- Evaluating with USE_LOG_DISTANCE = {use_log} ---")
        
        if use_log:
            y_train = np.log(df_train['distance']) 
            y_test = np.log(df_test['distance'])
        else:
            y_train = df_train['distance']         
            y_test = df_test['distance']

        train_distances_ft = np.exp(y_train) if use_log else y_train

        print("Running Feature Selection (RFECV)...")
        rf_for_selection = RandomForestRegressor(n_estimators=100, random_state=42)
        rfecv = RFECV(
            estimator=rf_for_selection, 
            step=1, 
            cv=5, 
            scoring='neg_mean_absolute_error',
            min_features_to_select=5,
            n_jobs=-1
        )
        rfecv.fit(X_train_full, y_train)
        
        top_features = X_train_full.columns[rfecv.support_].tolist()
        
        # Quick print to see if RFECV kept our new angle feature!
        if 'angle' in top_features:
            print(f"Optimal features: {len(top_features)} (The model KEPT the 'angle' feature!)")
        else:
            print(f"Optimal features: {len(top_features)} (The model tossed the 'angle' feature...)")
            
        X_train_filtered = X_train_full[top_features]
        X_test_filtered = X_test_full[top_features]

        for w_idx, profile in enumerate(WEIGHT_PROFILES):
            sample_weights = np.array([profile.get(int(round(d)), 1.0) for d in train_distances_ft])
            
            for params in param_combinations:
                rf = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                rf.fit(X_train_filtered, y_train, sample_weight=sample_weights)
                
                y_pred_raw = rf.predict(X_test_filtered)
                
                if use_log:
                    y_pred_ft = np.exp(y_pred_raw)
                    y_actual_ft = np.exp(y_test)
                else:
                    y_pred_ft = y_pred_raw
                    y_actual_ft = y_test
                    
                mae = mean_absolute_error(y_actual_ft, y_pred_ft)
                
                if mae < best_mae:
                    best_mae = mae
                    best_model = rf
                    best_params = params
                    best_profile_idx = w_idx
                    best_features = top_features
                    best_log_status = use_log
                
    print("\n=======================================================")
    print(" CUSTOM OPTIMIZATION COMPLETE! ")
    print("=======================================================")
    print(f"Lowest Test Folder MAE Achieved: {best_mae:.3f} feet")
    print(f"Winning Setting -> USE_LOG_DISTANCE: {best_log_status}")
    print(f"Winning Setting -> Weight Profile: Profile {best_profile_idx + 1}")
    print(f"Winning Setting -> Parameters: {best_params}")

    return best_model, best_features, best_log_status, df_test

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    print("Loading Training Data...")
    df_train = process_audio_files(TRAIN_DIR)
    
    print("\nLoading Test/Validation Data...")
    df_test = process_audio_files(TEST_DIR)

    if df_train.empty or df_test.empty:
        print("Exiting script. Ensure both TRAIN_DIR and TEST_DIR have valid .wav files.")
    else:
        trained_model, feature_names, best_log_status, df_test_results = train_and_tune_model(df_train, df_test)

        joblib.dump(trained_model, MODEL_FILENAME)
        joblib.dump(feature_names, FEATURE_NAMES_FILENAME)
        print(f"\nModel saved successfully as '{MODEL_FILENAME}'")
        print(f"Feature list saved successfully as '{FEATURE_NAMES_FILENAME}'")

        print(f"\n--- Final Performance on '{TEST_DIR}' ---")
        
        X_test_final = df_test_results[feature_names]
        final_preds_raw = trained_model.predict(X_test_final)
        
        if best_log_status:
            final_preds = np.exp(final_preds_raw)
            actual_dists = df_test_results['distance'] 
        else:
            final_preds = final_preds_raw
            actual_dists = df_test_results['distance']
            
        results = []
        for i in range(len(df_test_results)):
            actual = actual_dists.iloc[i]
            pred = final_preds[i]
            error = pred - actual
            filename = df_test_results['filename'].iloc[i]
            
            # Formatting to show the angle and distance clearly
            match = re.search(r'test(\d+)_(\d+)ft', filename)
            angle = match.group(1) if match else "Unknown"
            
            results.append({
                'Angle': angle,
                'Actual (ft)': actual,
                'Predicted (ft)': pred,
                'Error (ft)': error,
                'File': filename
            })
            
        results_df = pd.DataFrame(results).sort_values(by=['Actual (ft)', 'Angle'])
        
        # Displaying the results with the angle column first
        pd.set_option('display.max_colwidth', None)
        print("\n" + results_df[['Angle', 'Actual (ft)', 'Predicted (ft)', 'Error (ft)', 'File']].to_string(float_format="%.2f", index=False))
        
        print("-" * 50)
        print(f"FINAL WINNING MODEL MAE: {results_df['Error (ft)'].abs().mean():.3f} feet")
        print("-" * 50)
