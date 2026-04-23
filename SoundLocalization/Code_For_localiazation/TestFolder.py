import os
import re
import glob
import librosa
import numpy as np
import pandas as pd
import joblib

# ==========================================
# CONFIGURATION
# ==========================================
# Change this to the folder containing the files you want to test
TEST_DIR = 'single' 

MODEL_FILENAME = 'distance_model_rawApril6_angle_try1.joblib'
FEATURE_NAMES_FILENAME = 'feature_names_rawApril6_angle_try1.joblib'

# Audio Processing Parameters (MUST MATCH TRAINING EXACTLY)
HOP_LENGTH = 512
FRAME_LENGTH = 2048
SAMPLE_RATE = 48000

# --- Toggles ---
USE_LOG_DISTANCE = False  # MUST MATCH THE SETTING USED DURING TRAINING

# ==========================================
# EXACT FEATURE EXTRACTION FROM TRAINING
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
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    for i in range(13):
        features[f'mfcc_mean_{i+1}'] = mfccs_mean[i]
        features[f'mfcc_std_{i+1}'] = mfccs_std[i]
        
    return features

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("==================================================")
    print(" BATCH AUDIO DISTANCE PREDICTION ENGINE")
    print("==================================================")

    # 1. Load Model and Features
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(FEATURE_NAMES_FILENAME):
        print(f"ERROR: Could not find '{MODEL_FILENAME}' or '{FEATURE_NAMES_FILENAME}'")
        print("Please ensure this script is in the same folder as your trained model.")
        return

    print("Loading model and required feature list...")
    model = joblib.load(MODEL_FILENAME)
    required_features = joblib.load(FEATURE_NAMES_FILENAME)
    print(f"Model loaded. Model expects {len(required_features)} specific features.\n")

    # 2. Find Audio Files
    wav_files = glob.glob(os.path.join(TEST_DIR, '*.wav'))
    if not wav_files:
        print(f"ERROR: No .wav files found in directory '{TEST_DIR}'.")
        return

    print(f"Found {len(wav_files)} files to process. Extracting features...\n")

    results = []

    # 3. Process Each File
    for filepath in wav_files:
        filename = os.path.basename(filepath)
        
        try:
            # Load Audio
            y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            
            # Extract exactly the same features as training
            features_dict = extract_features(y, sr)
            features_df = pd.DataFrame([features_dict])
            
            # Filter down to just the mathematically perfect features found by RFECV
            features_df = features_df[required_features] 
            
            # Predict
            predicted_val = model.predict(features_df)[0]
            
            # Reverse the "Physics Hack" if the model was trained with it
            if USE_LOG_DISTANCE:
                predicted_dist_ft = np.exp(predicted_val) 
            else:
                predicted_dist_ft = predicted_val
            
            # Try to grab the actual distance from the filename (e.g., "10ft")
            match = re.search(r'(\d+)ft', filename)
            
            if match:
                actual_dist = float(match.group(1))
                error = predicted_dist_ft - actual_dist
                results.append({
                    'File': filename,
                    'Actual (ft)': actual_dist,
                    'Predicted (ft)': predicted_dist_ft,
                    'Error (ft)': error
                })
            else:
                # If no distance in filename, just output prediction
                results.append({
                    'File': filename,
                    'Actual (ft)': np.nan,
                    'Predicted (ft)': predicted_dist_ft,
                    'Error (ft)': np.nan
                })
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 4. Display Results
    if results:
        df_results = pd.DataFrame(results)
        
        # Sort by Actual Distance if available, otherwise by filename
        if df_results['Actual (ft)'].notna().any():
            df_results = df_results.sort_values(by='Actual (ft)')
        else:
            df_results = df_results.sort_values(by='File')

        print(df_results.to_string(float_format="%.2f", index=False))
        
        # Calculate MAE if we have actual distances
        if df_results['Error (ft)'].notna().any():
            mae = df_results['Error (ft)'].abs().mean()
            print("-" * 50)
            print(f"OVERALL MEAN ABSOLUTE ERROR: {mae:.2f} feet")
            print("-" * 50)

if __name__ == "__main__":
    main()