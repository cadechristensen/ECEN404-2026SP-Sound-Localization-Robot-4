import os
import sys
import warnings
import contextlib
import numpy as np
import pandas as pd
import torch
import joblib
import librosa
import doanet_model
import doanet_parameters
import cls_feature_class

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class AudioInferenceEngine:
    def __init__(self, task_id='6', models_dir='models'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = self._load_params(task_id)        
        self.seld_checkpoint = os.path.join(models_dir, "6_newdata_mic_dev_split1_model.h5")
        self.dist_model_path = 'distance_model_v1.joblib'
        self.dist_names_path = 'feature_names.joblib'        
        self.model_seld = self._load_seld_model()
        self.scaler_seld = self._load_scaler()
        self.model_dist, self.dist_feature_names = self._load_dist_model()
        print("System Ready.")

    def _load_params(self, task_id):
        with contextlib.redirect_stdout(None):
            params = doanet_parameters.get_params(task_id)
        params['nb_cnn2d_filt'] = 128
        params['rnn_size'] = 256
        params['self_attn'] = True
        params['unique_classes'] = 2
        return params

    def _load_seld_model(self):
        nb_ch = 10
        data_in = (self.params['batch_size'], nb_ch, self.params['feature_sequence_length'], self.params['nb_mel_bins'])
        data_out = [self.params['batch_size'], self.params['label_sequence_length'], self.params['unique_classes'] * 3]
        model = doanet_model.CRNN(data_in, data_out, self.params).to(self.device)
        model.eval()
        
        if os.path.exists(self.seld_checkpoint):
            model.load_state_dict(torch.load(self.seld_checkpoint, map_location=self.device))
        else:
            sys.exit(f"Error: SELD Model {self.seld_checkpoint} not found")
            
        return model

    def _load_scaler(self):
        feat_cls = cls_feature_class.FeatureClass(self.params)
        wts_file = feat_cls.get_normalized_wts_file()
        if not os.path.exists(wts_file):
             wts_file = os.path.join(self.params['feat_label_dir'], 'mic_wts')
        if not os.path.exists(wts_file): 
            sys.exit("Normalization weights not found")
        return joblib.load(wts_file)

    def _load_dist_model(self):
        if not os.path.exists(self.dist_model_path): 
            sys.exit("Distance model missing")
        model = joblib.load(self.dist_model_path)
        names = joblib.load(self.dist_names_path)
        return model, names

    def process_file(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return f"Error: File {filepath} not found."

        res_str = []        
        try:
            y_check, sr_check = librosa.load(filepath, sr=None, mono=False)
            if sr_check != self.params['fs']:
                y_check = librosa.resample(y_check, orig_sr=sr_check, target_sr=self.params['fs'], res_type='scipy')            
            feat_extractor = cls_feature_class.FeatureClass(self.params)
            features_SELD = feat_extractor.extract_features_for_file(filepath)
            features_SELD = self.scaler_seld.transform(features_SELD)            
            feat_seq_len = self.params['feature_sequence_length']
            nb_feat_frames = features_SELD.shape[0]
            batch_size_feat = int(np.ceil(nb_feat_frames / float(feat_seq_len)))
            feat_pad_len = batch_size_feat * feat_seq_len - nb_feat_frames
            if feat_pad_len > 0:
                features_SELD = np.pad(features_SELD, ((0, feat_pad_len), (0, 0)), 'constant', constant_values=1e-6)
            nb_ch = 10 
            features_SELD = features_SELD.reshape((batch_size_feat, feat_seq_len, nb_ch, self.params['nb_mel_bins']))
            features_SELD = np.transpose(features_SELD, (0, 2, 1, 3)) 
            data_SELD = torch.tensor(features_SELD).to(self.device).float()            
            output, activity_out = self.model_seld(data_SELD)
            max_nb_doas = output.shape[2] // 3
            output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
            output = output.view(-1, output.shape[-2], output.shape[-1])
            activity_out = activity_out.view(-1, activity_out.shape[-1])
            output = output.cpu().detach().numpy()
            sigmoid_scores = torch.sigmoid(activity_out).cpu().detach().numpy()            
            real_samples = y_check.shape[1] if y_check.ndim > 1 else y_check.shape[0]
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
                    x, y = output_real[mask, i, 0], output_real[mask, i, 1]
                    deg = np.degrees(np.arctan2(np.mean(y), np.mean(x))) % 360
                    conf = np.mean(sigmoid_scores_real[mask, i])
                    res_str.append(f"Source {i}: {deg:.1f}° (Loudness: {conf:.2f})")               
        except Exception as e:
            res_str.append(f"SELD Error: {e}")
        dist = predict_distance_from_file(
            self.model_dist, 
            self.dist_feature_names, 
            filepath, 
            sample_rate=48000,
            frame_length=2048,
            hop_length=256
        )
        if dist is not None:
            res_str.append(f"Distance: {dist:.1f} ft")
        
        return " | ".join(res_str) if res_str else "No active sources detected."

def extract_features(y, sr, frame_length, hop_length):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    spec_cent_mean = np.mean(spec_cent)
    spec_cent_std = np.std(spec_cent)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    features = {
        'rms_mean': rms_mean, 'rms_std': rms_std,
        'spec_cent_mean': spec_cent_mean, 'spec_cent_std': spec_cent_std,
    }
    for i in range(13):
        features[f'mfcc_mean_{i+1}'] = mfccs_mean[i]
    for i in range(13):
        features[f'mfcc_std_{i+1}'] = mfccs_std[i]

    return features

def predict_distance_from_file(model, feature_names, filepath, sample_rate, frame_length, hop_length):
    try:
        y, sr = librosa.load(filepath, sr=sample_rate)
        features_dict = extract_features(y, sr, frame_length, hop_length)
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[feature_names]  
        prediction = model.predict(features_df)
        return prediction[0]
    except Exception:
        return None

