import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
import librosa
import doanet_model
import doanet_parameters
import cls_feature_class
import pyaudio
import wave

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class Infer:
    def __init__(self, task_id='6', models_dir='models'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = doanet_parameters.get_params(task_id)
        self.params.update({
            'nb_cnn2d_filt': 128, 'rnn_size': 256, 
            'self_attn': True, 'unique_classes': 2
        })
        nb_ch = 10 
        self.seld_ch = nb_ch 
        
        data_in = (self.params['batch_size'], nb_ch, self.params['feature_sequence_length'], self.params['nb_mel_bins'])
        data_out = [self.params['batch_size'], self.params['label_sequence_length'], self.params['unique_classes'] * 3]
        
        self.model_seld = doanet_model.CRNN(data_in, data_out, self.params).to(self.device)
        self.model_seld.eval()
        
        seld_ckpt = os.path.join(models_dir, "6_newdata_mic_dev_split1_model.h5")
        if os.path.exists(seld_ckpt):
            self.model_seld.load_state_dict(torch.load(seld_ckpt, map_location=self.device))
        else:
            sys.exit(f"Error: Model {seld_ckpt} not found")
        feat_cls = cls_feature_class.FeatureClass(self.params)
        wts_file = feat_cls.get_normalized_wts_file()
        if not os.path.exists(wts_file):
             wts_file = os.path.join(self.params['feat_label_dir'], 'mic_wts')
        if not os.path.exists(wts_file): 
            sys.exit("Normalization weights not found")
        self.scaler_seld = joblib.load(wts_file)

        dist_model_path = 'distance_model_v1.joblib'
        if not os.path.exists(dist_model_path): 
            sys.exit("Distance model missing")
        self.model_dist = joblib.load(dist_model_path)
        self.dist_feature_names = joblib.load('feature_names.joblib')
        
        print("System Ready.")

    def _extract_dist_features(self, y, sr, frame_length=2048, hop_length=256):
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        features = {
            'rms_mean': np.mean(rms), 'rms_std': np.std(rms),
            'spec_cent_mean': np.mean(spec_cent), 'spec_cent_std': np.std(spec_cent),
        }
        features.update({f'mfcc_mean_{i+1}': m for i, m in enumerate(mfccs_mean)})
        features.update({f'mfcc_std_{i+1}': s for i, s in enumerate(mfccs_std)})
        return features

    def process_file(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return f"Error: File {filepath} not found."

        res_str = []
        try:
            y_raw, sr_raw = librosa.load(filepath, sr=48000, mono=False)
            
            if self.params['fs'] != 48000:
                y_seld = librosa.resample(y_raw, orig_sr=48000, target_sr=self.params['fs'], res_type='scipy')
            else:
                y_seld = y_raw

            feat_extractor = cls_feature_class.FeatureClass(self.params)
            features_SELD = feat_extractor.extract_features_for_file(filepath)
            features_SELD = self.scaler_seld.transform(features_SELD)
            
            feat_seq_len = self.params['feature_sequence_length']
            nb_feat_frames = features_SELD.shape[0]
            batch_size_feat = int(np.ceil(nb_feat_frames / float(feat_seq_len)))
            feat_pad_len = batch_size_feat * feat_seq_len - nb_feat_frames         
            if feat_pad_len > 0:
                features_SELD = np.pad(features_SELD, ((0, feat_pad_len), (0, 0)), 'constant', constant_values=1e-6)
            
            features_SELD = features_SELD.reshape((batch_size_feat, feat_seq_len, self.seld_ch, self.params['nb_mel_bins']))
            features_SELD = np.transpose(features_SELD, (0, 2, 1, 3)) 
            data_SELD = torch.tensor(features_SELD).to(self.device).float()
            
            output, activity_out = self.model_seld(data_SELD)
            
            max_nb_doas = output.shape[2] // 3
            output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
            output = output.view(-1, output.shape[-2], output.shape[-1])
            activity_out = activity_out.view(-1, activity_out.shape[-1])
            
            output = output.cpu().detach().numpy()
            sigmoid_scores = torch.sigmoid(activity_out).cpu().detach().numpy()
            
            real_samples = y_seld.shape[1] if y_seld.ndim > 1 else y_seld.shape[0]
            hop_len_samples = self.params['hop_len_s'] * self.params['fs']
            hop_ratio = self.params['label_hop_len_s'] / self.params['hop_len_s']
            nb_feat_frames_real = int(np.ceil(real_samples / float(hop_len_samples)))
            nb_label_frames = int(np.ceil(nb_feat_frames_real / hop_ratio))
            
            output_real = output[:nb_label_frames]
            sigmoid_scores_real = sigmoid_scores[:nb_label_frames]
            activity_real = (sigmoid_scores_real > 0.4)
            
            for i in range(2): 
                mask = activity_real[:, i]
                if np.any(mask):
                    x, y_coords = output_real[mask, i, 0], output_real[mask, i, 1]
                    deg = np.degrees(np.arctan2(np.mean(y_coords), np.mean(x))) % 360
                    conf = np.mean(sigmoid_scores_real[mask, i])
                    res_str.append(f"Source {i}: {deg:.1f}° (Loudness: {conf:.2f})")

            if y_raw.ndim > 1:
                y_dist = np.mean(y_raw, axis=0)
            else:
                y_dist = y_raw

            features_dict = self._extract_dist_features(y_dist, sr=48000)
            features_df = pd.DataFrame([features_dict])[self.dist_feature_names]
            dist_pred = self.model_dist.predict(features_df)[0]
            res_str.append(f"Distance: {dist_pred:.1f} ft")

        except Exception as e:
            res_str.append(f"Inference Error: {e}")

        return " | ".join(res_str) if res_str else "No active sources detected."

def record_audio(filename: str = "live_input.wav", duration: float = 10.0, 
                         sample_rate: int = 48000, channels: int = 4, 
                         chunk_size: int = 1024) -> str:
    audio = pyaudio.PyAudio()
    stream = None
    try:
        device_index = None
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] >= channels:
                device_index = i
                break
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )
        print(f"Recording {duration}s...", end="", flush=True)
        frames = []
        total_chunks = int(sample_rate / chunk_size * duration)
        for _ in range(total_chunks):
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
        print(" Done.")
        stream.stop_stream()
        stream.close()        
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.reshape(-1, channels).flatten()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2) 
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        return filename

    except Exception as e:
        print(f"\nRecording Error: {e}")
        return None
        
    finally:
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        audio.terminate()