import os
import sys
import time
import warnings
import contextlib
import argparse
import wave
import numpy as np
import pyaudio

# --- 1. SILENCE WARNINGS & LAZY SETUP ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Global Heavy Imports (Loaded later)
import torch
import joblib
import librosa
import doanet_model
import doanet_parameters
import cls_feature_class

# --- 2. AUDIO RECORDER CLASS ---
class AudioRecorder:
    def __init__(self, sample_rate: int = 48000, channels: int = 4,
                 chunk_size: int = 1024, audio_format: int = pyaudio.paInt16,
                 record_channels: int = 8):
        self.sample_rate = sample_rate
        self.channels = channels  # Channels to save (4 for SELD)
        self.record_channels = record_channels  # Channels to capture (8 for TI Board)
        self.chunk_size = chunk_size
        self.audio_format = audio_format
        self.audio = None

    def list_devices(self):
        audio = pyaudio.PyAudio()
        print("\nAvailable Audio Devices:")
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} (Ch: {info['maxInputChannels']}, Rate: {int(info['defaultSampleRate'])})")
        audio.terminate()

    def record_audio(self, filename: str, duration: float, device_index: int = None) -> bool:
        try:
            self.audio = pyaudio.PyAudio()
            
            stream_kwargs = {
                'format': self.audio_format,
                'channels': self.record_channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size
            }

            if device_index is not None:
                stream_kwargs['input_device_index'] = device_index

            stream = self.audio.open(**stream_kwargs)

            print(f"\nListening for {duration}s...", end="", flush=True)
            
            frames = []
            total_chunks = int(self.sample_rate / self.chunk_size * duration)

            for _ in range(total_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)

            print(" Done.")
            
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

            # Process: Extract 4 channels from 8
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.reshape(-1, self.record_channels)
            
            # Keep first 'channels' (4)
            audio_data = audio_data[:, :self.channels]
            
            # Flatten and save
            audio_data = audio_data.flatten()

            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            return True

        except Exception as e:
            print(f"\nRecording Error: {e}")
            if self.audio: self.audio.terminate()
            return False

# --- 3. INFERENCE FUNCTIONS ---

def extract_features_dict_safe(y, sr, frame_length, hop_length):
    if y.ndim > 1:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y

    rms = librosa.feature.rms(y=y_mono, frame_length=frame_length, hop_length=hop_length)
    spec_cent = librosa.feature.spectral_centroid(y=y_mono, sr=sr, n_fft=frame_length, hop_length=hop_length)
    mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)

    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)

    features = {
        'rms_mean': np.mean(rms), 'rms_std': np.std(rms),
        'spec_cent_mean': np.mean(spec_cent), 'spec_cent_std': np.std(spec_cent),
    }
    for i in range(13):
        features[f'mfcc_mean_{i+1}'] = mfcc_means[i]
        features[f'mfcc_std_{i+1}'] = mfcc_stds[i]

    return features

def predict_distance(model, feature_names, y, sr):
    try:
        DIST_FRAME = 2048
        DIST_HOP = 256
        feat_dict = extract_features_dict_safe(y, sr, DIST_FRAME, DIST_HOP)
        feat_vector = [feat_dict[name] for name in feature_names]
        prediction = model.predict([feat_vector])
        return prediction[0]
    except Exception:
        return None

def run_inference(filepath, params, device, model_SELD, spec_scaler, model_dist, dist_names):
    try:
        # Critical: mono=False for SELD
        y_test, sr_test = librosa.load(filepath, sr=None, mono=False)
        
        if sr_test != params['fs']:
            y_test = librosa.resample(y_test, orig_sr=sr_test, target_sr=params['fs'], res_type='linear')
    except Exception as e:
        return f"Error loading audio: {e}"

    # SELD Extraction
    feature_extractor = cls_feature_class.FeatureClass(params)
    # Use the file extraction to be safe/compatible
    features_SELD = feature_extractor.extract_features_for_file(filepath) 
    features_SELD = spec_scaler.transform(features_SELD)

    # Pad & Reshape
    feat_seq_len = params['feature_sequence_length']
    nb_feat_frames = features_SELD.shape[0]
    batch_size_feat = int(np.ceil(nb_feat_frames / float(feat_seq_len)))
    feat_pad_len = batch_size_feat * feat_seq_len - nb_feat_frames
    if feat_pad_len > 0:
        features_SELD = np.pad(features_SELD, ((0, feat_pad_len), (0, 0)), 'constant', constant_values=1e-6)

    features_SELD = features_SELD.reshape((batch_size_feat, feat_seq_len, -1, params['nb_mel_bins']))
    features_SELD = np.transpose(features_SELD, (0, 2, 1, 3))
    data_SELD = torch.tensor(features_SELD).to(device).float()

    # Prediction
    output, activity_out = model_SELD(data_SELD)

    # Post-Process
    max_nb_doas = output.shape[2] // 3
    output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
    output = output.reshape(-1, output.shape[-2], output.shape[-1])
    activity_out = activity_out.reshape(-1, activity_out.shape[-1])
    
    output_np = output.cpu().detach().numpy()
    scores_np = torch.sigmoid(activity_out).cpu().detach().numpy()
    
    # Real frame logic
    real_samples = y_test.shape[1] if y_test.ndim > 1 else y_test.shape[0]
    hop_ratio = params['label_hop_len_s'] / params['hop_len_s']
    nb_label_frames = int(np.ceil((real_samples / (params['hop_len_s'] * params['fs'])) / hop_ratio))
    
    output_real = output_np[:nb_label_frames]
    scores_real = scores_np[:nb_label_frames]
    activity_real = (scores_real > 0.4)

    # Format
    res_str = []
    for i in range(2):
        mask = activity_real[:, i]
        if np.any(mask):
            active_x = output_real[mask, i, 0]
            active_y = output_real[mask, i, 1]
            azimuth = np.degrees(np.arctan2(np.mean(active_y), np.mean(active_x))) % 360
            score = np.mean(scores_real[mask, i])
            res_str.append(f"Src {i}: {azimuth:.1f}° (Conf: {score:.2f})")
    
    if not res_str:
        res_str.append("No active sources detected.")

    # Distance
    dist_pred = predict_distance(model_dist, dist_names, y_test, params['fs'])
    if dist_pred:
        res_str.append(f"Dist: {dist_pred:.1f} ft")

    return " | ".join(res_str)

# --- 4. MAIN LOOP ---

def main():
    parser = argparse.ArgumentParser(description='Live SELD Inference')
    
    # FIX: Make 'task' a positional argument so "python script.py 6" works
    parser.add_argument('task', nargs='?', default='1', help='Task ID (e.g. 6)')
    
    parser.add_argument('--device', type=int, default=None, help='Audio input device index')
    parser.add_argument('--duration', type=float, default=3.0, help='Recording duration (seconds)')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    
    args = parser.parse_args()

    recorder = AudioRecorder(record_channels=8, channels=4)

    if args.list_devices:
        recorder.list_devices()
        return
    
    print("Initializing AI Engine (approx 5-10s)...")

    # Load Params
    task_id = args.task
    with contextlib.redirect_stdout(None):
        params = doanet_parameters.get_params(task_id)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load SELD
    if params['dataset'] == 'mic': nb_ch = 10
    else: nb_ch = 7 
    
    data_in = (params['batch_size'], nb_ch, params['feature_sequence_length'], params['nb_mel_bins'])
    data_out = [params['batch_size'], params['label_sequence_length'], params['unique_classes'] * 3]
    
    checkpoint_name = "models/6_newdata_mic_dev_split1_model.h5"
    if not os.path.exists(checkpoint_name):
        sys.exit(f"Error: SELD model not found at {checkpoint_name}")

    model_SELD = doanet_model.CRNN(data_in, data_out, params).to(device)
    model_SELD.eval()
    model_SELD.load_state_dict(torch.load(checkpoint_name, map_location=device))

    # Load Scaler
    feat_cls = cls_feature_class.FeatureClass(params)
    wts_file = feat_cls.get_normalized_wts_file()
    if not os.path.exists(wts_file):
         wts_file = os.path.join(params['feat_label_dir'], os.path.basename(wts_file))
    spec_scaler = joblib.load(wts_file)

    # Load Distance
    DIST_MODEL = 'distance_model_v1.joblib'
    DIST_NAMES = 'feature_names.joblib'
    if not os.path.exists(DIST_MODEL): sys.exit("Distance model missing")
    model_dist = joblib.load(DIST_MODEL)
    dist_names = joblib.load(DIST_NAMES)

    print("\n=== SYSTEM READY ===")
    print(f"Using Task {task_id} | Duration: {args.duration}s")
    
    temp_filename = "live_temp.wav"

    try:
        while True:
            user_in = input("\nPress ENTER to record (or 'q' to quit): ")
            if user_in.lower() == 'q':
                break
            
            success = recorder.record_audio(temp_filename, duration=args.duration, device_index=args.device)
            
            if success:
                result = run_inference(temp_filename, params, device, model_SELD, spec_scaler, model_dist, dist_names)
                print(f"RESULT: {result}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except: pass

if __name__ == "__main__":
    main()