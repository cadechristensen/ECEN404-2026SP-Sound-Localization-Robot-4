import os
import scipy.io.wavfile as wav

# Point this to your audio folder based on your console output
aud_dir = 'DCASE2020_SELD_dataset/mic_dev'

print(f"Scanning {aud_dir} for corrupted files...")
corrupted_files = []

for file_name in os.listdir(aud_dir):
    if not file_name.endswith('.wav') or file_name.startswith('._'):
        continue
    
    file_path = os.path.join(aud_dir, file_name)
    try:
        # Try to read the file
        fs, audio = wav.read(file_path)
    except Exception as e:
        print(f"🚨 CORRUPT FILE FOUND: {file_name} -> {e}")
        corrupted_files.append(file_name)

print("="*50)
if len(corrupted_files) == 0:
    print("✅ All files are perfectly healthy!")
else:
    print(f"❌ Found {len(corrupted_files)} corrupted file(s).")
    print("Please delete these .wav files AND their matching .csv files from metadata_dev.")