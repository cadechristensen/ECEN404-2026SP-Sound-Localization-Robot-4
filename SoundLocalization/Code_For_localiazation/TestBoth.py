import numpy as np
import os
import sys
import cls_data_generator
import doanet_model
import doanet_parameters
import torch
from IPython import embed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
plot.rcParams.update({'font.size': 22})
import joblib 
import cls_feature_class
import librosa
import pandas as pd
import re
DISTANCE_MODEL_FILENAME = 'distance_model_v1.joblib'
FEATURE_NAMES_FILENAME = 'feature_names.joblib'


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
    except Exception as e:
        print(f"Error predicting distance for file {filepath}: {e}")
        return None


def main(argv):


    wav_file_path = 'TestDat/2026-03-23_19-48-48_test135deg_raw.wav' 
    #wav_file_path = 'TestDat/2026-03-20_19-19-47_test_180deg_filtered.wav' 
    #wav_file_path = 'TestDat/180degtest.wav' 
    label_npy_filename = 'fold7_room1_mix001.npy' 
    checkpoint_name = "models/6_modelwithmore_mic_dev_split1_model.h5"
    #checkpoint_name = "models/6_newmodelflat_mic_dev_split1_model.h5"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    task_id = '1' if len(argv) < 2 else argv[1]
    params = doanet_parameters.get_params(task_id)

    
    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=7, shuffle=False, is_eval=False 
    )
    
    if params['dataset'] == 'foa':
        nb_ch = 4 + 3
    elif params['dataset'] == 'mic':
        nb_ch = 4 + 6
    else:
        raise ValueError(f"Unknown dataset type: {params['dataset']}")

    data_in = (params['batch_size'], nb_ch, params['feature_sequence_length'], params['nb_mel_bins'])
    data_out = [params['batch_size'], params['label_sequence_length'], params['unique_classes'] * 3]
    
    
    dump_figures = True

    model_SELD = doanet_model.CRNN(data_in, data_out, params).to(device)
    model_SELD.eval()
    model_SELD.load_state_dict(torch.load(checkpoint_name, map_location=torch.device('cpu')))
    
    if dump_figures:
        dump_folder = "Results" 
        os.makedirs(dump_folder, exist_ok=True)


    if not os.path.exists(DISTANCE_MODEL_FILENAME):
        print("Distance model not found: {DISTANCE_MODEL_FILENAME}")
        sys.exit(1)
    model_distance = joblib.load(DISTANCE_MODEL_FILENAME)
    
    if not os.path.exists(FEATURE_NAMES_FILENAME):
        print(f"Feature names not found: {FEATURE_NAMES_FILENAME}")
        sys.exit(1)
    distance_feature_names = joblib.load(FEATURE_NAMES_FILENAME)

    
    with torch.no_grad():
        y_test, sr_test = librosa.load(wav_file_path, sr=None) 
        if sr_test != params['fs']:
            y_test = librosa.resample(y_test, orig_sr=sr_test, target_sr=params['fs'])
        
        real_samples = y_test.shape[0]
        
        hop_len_samples = params['hop_len_s'] * params['fs']
        hop_ratio = params['label_hop_len_s'] / params['hop_len_s']
        
        nb_feat_frames_real = int(np.ceil(real_samples / float(hop_len_samples)))
        wav_label_frames = int(np.ceil(nb_feat_frames_real / hop_ratio))

        feature_extractor_SELD = cls_feature_class.FeatureClass(params)
        wts_file = feature_extractor_SELD.get_normalized_wts_file()
        if not os.path.exists(wts_file):
            wts_file = os.path.join(params['feat_label_dir'], os.path.basename(wts_file))
            if not os.path.exists(wts_file):
                print(f"Also not found at: {wts_file}")
                sys.exit(1)
            
                
        spec_scaler_SELD = joblib.load(wts_file)
        
        if not os.path.exists(wav_file_path):
             print(f"WAV file not found: {wav_file_path}")
             sys.exit(1)
        features_SELD = feature_extractor_SELD.extract_features_for_file(wav_file_path)
        features_SELD = spec_scaler_SELD.transform(features_SELD)
        
        label_file_path = os.path.join(data_gen_test._label_dir, label_npy_filename)
        if not os.path.exists(label_file_path):
             print(f"Label file not found: {label_file_path}")
             sys.exit(1)
        target_data = np.load(label_file_path)
        
        feat_seq_len = params['feature_sequence_length']
        label_seq_len = params['label_sequence_length']
        
        nb_feat_frames = features_SELD.shape[0] 
        batch_size_feat = int(np.ceil(nb_feat_frames / float(feat_seq_len)))
        feat_pad_len = batch_size_feat * feat_seq_len - nb_feat_frames
        if feat_pad_len > 0:
            features_SELD = np.pad(features_SELD, ((0, feat_pad_len), (0, 0)), 'constant', constant_values=1e-6)

        nb_label_frames = wav_label_frames 
        npy_label_frames = target_data.shape[0] 

        batch_size_label = int(np.ceil(npy_label_frames / float(label_seq_len)))
        label_pad_len = batch_size_label * label_seq_len - npy_label_frames
        if label_pad_len > 0:
            target_data = np.pad(target_data, ((0, label_pad_len), (0, 0)), 'constant', constant_values=0)

        if nb_label_frames > target_data.shape[0]:
            extra_pad = nb_label_frames - target_data.shape[0]
            target_data = np.pad(target_data, ((0, extra_pad), (0, 0)), 'constant', constant_values=0)

        features_SELD = features_SELD.reshape((batch_size_feat, feat_seq_len, features_SELD.shape[1]))
        features_SELD = features_SELD.reshape((batch_size_feat, feat_seq_len, nb_ch, params['nb_mel_bins']))
        features_SELD = np.transpose(features_SELD, (0, 2, 1, 3)) 
        
        target_data = target_data.reshape((batch_size_label, label_seq_len, target_data.shape[1]))
        
        data_SELD = torch.tensor(features_SELD).to(device).float()
        target = torch.tensor(target_data[:,:,:-params['unique_classes']]).to(device).float() 
        
        output, activity_out = model_SELD(data_SELD)

        max_nb_doas = output.shape[2]//3
        output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
        target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)
        output, target, activity_out = output.view(-1, output.shape[-2], output.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1]), activity_out.view(-1, activity_out.shape[-1])
        output_norm = torch.sqrt(torch.sum(output**2, -1) + 1e-10)
        output = output/output_norm.unsqueeze(-1)
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        use_activity_detector = True
        activity_threshold = 0.4 
        
        sigmoid_scores = torch.sigmoid(activity_out).cpu().detach().numpy()

        if use_activity_detector:
            activity = (sigmoid_scores > activity_threshold)

        mel_spec = data_SELD[0][0].cpu()
        
        if params['dataset'] == 'foa':
            foa_iv = data_SELD[0][-1].cpu() 
        elif params['dataset'] == 'mic':
            foa_iv = data_SELD[0][4].cpu()
        else:
            foa_iv = data_SELD[0][-1].cpu()
            
        target[target > 1] = 0 

        
        time_axis = np.arange(0, nb_label_frames) * params['label_hop_len_s']
        
        plot.figure(figsize=(20,10))
        plot.subplot(321), plot.imshow(torch.transpose(mel_spec, -1, -2))
        plot.title("Mel Spectrogram")
        
        plot.subplot(322), plot.imshow(torch.transpose(foa_iv, -1, -2))
        plot.title("IV / GCC Features")
        
        plot.subplot(323), plot.plot(time_axis, target[:nb_label_frames, 0, 0], 'r', lw=2, label='X')
        plot.subplot(323), plot.plot(time_axis, target[:nb_label_frames, 0, 1], 'g', lw=2, label='Y')
        plot.subplot(323), plot.plot(time_axis, target[:nb_label_frames, 0, 2], 'b', lw=2, label='Z')
        plot.grid()
        plot.ylim([-1.1, 1.1])
        plot.title("Ground Truth XYZ - Source 0")
        plot.legend(loc='upper right') 
        plot.xlabel('Time (seconds)') 

        plot.subplot(324), plot.plot(time_axis, target[:nb_label_frames, 1, 0], 'r', lw=2, label='X')
        plot.subplot(324), plot.plot(time_axis, target[:nb_label_frames, 1, 1], 'g', lw=2, label='Y')
        plot.subplot(324), plot.plot(time_axis, target[:nb_label_frames, 1, 2], 'b', lw=2, label='Z')
        plot.grid()
        plot.ylim([-1.1, 1.1])
        plot.title("Ground Truth XYZ - Source 1")
        plot.legend(loc='upper right') 
        plot.xlabel('Time (seconds)') 

        if use_activity_detector:
            output[:, 0, 0:3] = activity[:, 0][:, np.newaxis]*output[:, 0, 0:3]
            output[:, 1, 0:3] = activity[:, 1][:, np.newaxis]*output[:, 1, 0:3]

        plot.subplot(325), plot.plot(time_axis, output[:nb_label_frames, 0, 0], 'r', lw=2, label='X')
        plot.subplot(325), plot.plot(time_axis, output[:nb_label_frames, 0, 1], 'g', lw=2, label='Y')
        plot.subplot(325), plot.plot(time_axis, output[:nb_label_frames, 0, 2], 'b', lw=2, label='Z')
        plot.grid()
        plot.ylim([-1.1, 1.1])
        plot.title("Predicted XYZ - Source 0")
        plot.legend(loc='upper right') 
        plot.xlabel('Time (seconds)') 

        plot.subplot(326), plot.plot(time_axis, output[:nb_label_frames, 1, 0], 'r', lw=2, label='X')
        plot.subplot(326), plot.plot(time_axis, output[:nb_label_frames, 1, 1], 'g', lw=2, label='Y')
        plot.subplot(326), plot.plot(time_axis, output[:nb_label_frames, 1, 2], 'b', lw=2, label='Z')
        plot.grid()
        plot.ylim([-1.1, 1.1])
        plot.title("Predicted XYZ - Source 1")
        plot.legend(loc='upper right') 
        plot.xlabel('Time (seconds)') 
        plot.tight_layout()

        if dump_figures:
            fig_name = '{}'.format(os.path.join(dump_folder, '{}.png'.format(os.path.basename(wav_file_path).split('.')[0])))
            print('saving figure : {}'.format(fig_name))
            plot.savefig(fig_name, dpi=100)
            plot.close()
        else:
            plot.show()

    output_real = output[:nb_label_frames]
    activity_real = activity[:nb_label_frames]
    sigmoid_scores_real = sigmoid_scores[:nb_label_frames]
    
    activity_mask_source0 = activity_real[:, 0]
    activity_mask_source1 = activity_real[:, 1]

    if np.any(activity_mask_source0): 
        active_x_0 = output_real[activity_mask_source0, 0, 0]
        active_y_0 = output_real[activity_mask_source0, 0, 1]
        
        mean_x_0 = np.mean(active_x_0)
        mean_y_0 = np.mean(active_y_0)
        
        mean_azimuth_rad_0 = np.arctan2(mean_y_0, mean_x_0)
        mean_azimuth_deg_0 = np.degrees(mean_azimuth_rad_0) % 360        
        active_scores_0 = sigmoid_scores_real[activity_mask_source0, 0]
        mean_score_0 = np.mean(active_scores_0)
        
        print(f"  Predicted Angle (Source 0): {mean_azimuth_deg_0:.1f} degrees")
        print(f"  Avg. Loudness Score (Source 0): {mean_score_0:.2f} (0.0-1.0)") 
    else:
        print("  Predicted Angle (Source 0): No activity detected.")

    if np.any(activity_mask_source1):
        active_x_1 = output_real[activity_mask_source1, 1, 0]
        active_y_1 = output_real[activity_mask_source1, 1, 1]
        
        mean_x_1 = np.mean(active_x_1)
        mean_y_1 = np.mean(active_y_1)
        mean_azimuth_rad_1 = np.arctan2(mean_y_1, mean_x_1)
        mean_azimuth_deg_1 = np.degrees(mean_azimuth_rad_1) % 360        
        active_scores_1 = sigmoid_scores_real[activity_mask_source1, 1]
        mean_score_1 = np.mean(active_scores_1)
        
        print(f"  Predicted Angle (Source 1): {mean_azimuth_deg_1:.1f} degrees")
        print(f"  Avg. Loudness Score (Source 1): {mean_score_1:.2f} (0.0-1.0)") 
    else:
        print("  Predicted Angle (Source 1): No activity detected.")


    
    DISTANCE_SAMPLE_RATE = 48000
    DISTANCE_FRAME_LENGTH = 2048
    DISTANCE_HOP_LENGTH = 256
    
    predicted_dist = predict_distance_from_file(
        model_distance, 
        distance_feature_names, 
        wav_file_path,
        DISTANCE_SAMPLE_RATE,
        DISTANCE_FRAME_LENGTH,
        DISTANCE_HOP_LENGTH
    )

    if predicted_dist is not None:
        match = re.search(r'(\d+)ft', os.path.basename(wav_file_path))
        print()
        if match:
            actual_dist = int(match.group(1))
            error = predicted_dist - actual_dist
            print(f"Actual Distance:      {actual_dist:.1f} ft")
            print(f"Predicted Distance:   {predicted_dist:.1f} ft")
            print(f"Error:                {error:.1f} ft")
        else:
            print(f"Predicted Distance:   {predicted_dist:.1f} ft")
        print()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)