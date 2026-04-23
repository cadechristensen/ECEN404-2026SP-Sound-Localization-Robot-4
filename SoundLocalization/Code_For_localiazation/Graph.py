import numpy as np
import os
import sys
import cls_data_generator # Re-added this import
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

def main(argv):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = doanet_parameters.get_params(task_id)

    print('\nLoading the best model and predicting results on the testing split')
    
    print('\tInitializing DataGenerator to get paths...')
    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=1, shuffle=False, is_eval=False 
    )
    
    if params['dataset'] == 'foa':
        nb_ch = 4 + 3  # 4 Mel + 3 Intensity Vectors
    elif params['dataset'] == 'mic':
        nb_ch = 4 + 6  # 4 Mel + 6 GCC-PHAT
    else:
        raise ValueError(f"Unknown dataset type: {params['dataset']}")

    data_in = (params['batch_size'], nb_ch, params['feature_sequence_length'], params['nb_mel_bins'])
    data_out = [params['batch_size'], params['label_sequence_length'], params['unique_classes'] * 3]
    
    print(f"\tCalculated data_in: {data_in}")
    print(f"\tCalculated data_out: {data_out}")
    
    dump_figures = True

    # CHOOSE THE MODEL WHOSE OUTPUT YOU WANT TO VISUALIZE 
    checkpoint_name = "models/6_newdata_mic_dev_split1_model.h5"
    model = doanet_model.CRNN(data_in, data_out, params)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_name, map_location=torch.device('cpu')))
    model = model.to(device)
    if dump_figures:
        dump_folder = os.path.join('dump_dir', os.path.basename(checkpoint_name).split('.')[0])
        os.makedirs(dump_folder, exist_ok=True)

    # ---- HARDCODE YOUR FILE PATHS HERE ----
    wav_file_path = 'TestDat/315test.wav' 
    label_npy_filename = 'fold7_room1_mix015.npy' # <--- ADD YOUR .NPY FILENAME
    # ---- END HARDCODING ----
    
    with torch.no_grad():
        print(f"Loading single .wav file: {wav_file_path}")
        print(f"Loading single .npy label file: {label_npy_filename}")

        # --- 1. Load Feature Extractor and Normalizer ---
        feature_extractor = cls_feature_class.FeatureClass(params)
        wts_file = feature_extractor.get_normalized_wts_file()
        if not os.path.exists(wts_file):
            wts_file = os.path.join(params['feat_label_dir'], os.path.basename(wts_file))
            if not os.path.exists(wts_file):
                print(f"ERROR: Also not found at: {wts_file}")
                sys.exit(1)
            else:
                print(f"Found weights at: {wts_file}")
                
        spec_scaler = joblib.load(wts_file)
        
        # --- 2. Extract and Normalize Features ---
        print("Extracting features from .wav file...")
        if not os.path.exists(wav_file_path):
             print(f"ERROR: WAV file not found: {wav_file_path}")
             sys.exit(1)
        features = feature_extractor.extract_features_for_file(wav_file_path)
        features = spec_scaler.transform(features)
        
        # --- 3. Load Labels (ADDED BACK) ---
        print("Loading labels from .npy file...")
        label_file_path = os.path.join(data_gen_test._label_dir, label_npy_filename)
        if not os.path.exists(label_file_path):
             print(f"ERROR: Label file not found: {label_file_path}")
             sys.exit(1)
        target_data = np.load(label_file_path)
        
        # --- 4. Replicate Batching/Padding ---
        feat_seq_len = params['feature_sequence_length']
        label_seq_len = params['label_sequence_length']
        
        nb_feat_frames = features.shape[0]
        batch_size_feat = int(np.ceil(nb_feat_frames / float(feat_seq_len)))
        feat_pad_len = batch_size_feat * feat_seq_len - nb_feat_frames
        if feat_pad_len > 0:
            features = np.pad(features, ((0, feat_pad_len), (0, 0)), 'constant', constant_values=1e-6)

        nb_label_frames = target_data.shape[0]
        batch_size_label = int(np.ceil(nb_label_frames / float(label_seq_len)))
        label_pad_len = batch_size_label * label_seq_len - nb_label_frames
        if label_pad_len > 0:
            target_data = np.pad(target_data, ((0, label_pad_len), (0, 0)), 'constant', constant_values=0)
            
        # --- 5. Reshape and Convert to Tensor ---
        features = features.reshape((batch_size_feat, feat_seq_len, features.shape[1]))
        features = features.reshape((batch_size_feat, feat_seq_len, nb_ch, params['nb_mel_bins']))
        features = np.transpose(features, (0, 2, 1, 3)) 
        
        target_data = target_data.reshape((batch_size_label, label_seq_len, target_data.shape[1]))
        
        data = torch.tensor(features).to(device).float()
        target = torch.tensor(target_data[:,:,:-params['unique_classes']]).to(device).float() 
        
        print("Running model prediction...")
        output, activity_out = model(data)

        max_nb_doas = output.shape[2]//3
        output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
        target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

        output, target, activity_out = output.view(-1, output.shape[-2], output.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1]), activity_out.view(-1, activity_out.shape[-1])
        output_norm = torch.sqrt(torch.sum(output**2, -1) + 1e-10)
        output = output/output_norm.unsqueeze(-1)

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        use_activity_detector = True
        if use_activity_detector:
            activity = (torch.sigmoid(activity_out).cpu().detach().numpy() > 0.25)
        mel_spec = data[0][0].cpu()
        
        if params['dataset'] == 'foa':
            foa_iv = data[0][-1].cpu() 
        elif params['dataset'] == 'mic':
            foa_iv = data[0][4].cpu()
        else:
            foa_iv = data[0][-1].cpu()
            
        target[target > 1] = 0 

        # --- PLOTTING IS 3x2 ---
        plot.figure(figsize=(20,10))
        plot.subplot(321), plot.imshow(torch.transpose(mel_spec, -1, -2))
        plot.title("Mel Spectrogram")
        
        plot.subplot(322), plot.imshow(torch.transpose(foa_iv, -1, -2))
        plot.title("IV / GCC Features")

        # --- Ground Truth Source 0 ---
        plot.subplot(323), plot.plot(target[:params['label_sequence_length'], 0, 0], 'r', lw=2, label='X')
        plot.subplot(323), plot.plot(target[:params['label_sequence_length'], 0, 1], 'g', lw=2, label='Y')
        plot.subplot(323), plot.plot(target[:params['label_sequence_length'], 0, 2], 'b', lw=2, label='Z')
        plot.grid()
        plot.ylim([-1.1, 1.1])
        plot.title("Ground Truth XYZ - Source 0")
        plot.legend(loc='upper right') # Added Key

        # --- Ground Truth Source 1 ---
        plot.subplot(324), plot.plot(target[:params['label_sequence_length'], 1, 0], 'r', lw=2, label='X')
        plot.subplot(324), plot.plot(target[:params['label_sequence_length'], 1, 1], 'g', lw=2, label='Y')
        plot.subplot(324), plot.plot(target[:params['label_sequence_length'], 1, 2], 'b', lw=2, label='Z')
        plot.grid()
        plot.ylim([-1.1, 1.1])
        plot.title("Ground Truth XYZ - Source 1")
        plot.legend(loc='upper right') # Added Key

        if use_activity_detector:
            output[:, 0, 0:3] = activity[:, 0][:, np.newaxis]*output[:, 0, 0:3]
            output[:, 1, 0:3] = activity[:, 1][:, np.newaxis]*output[:, 1, 0:3]

        # --- Predicted Source 0 ---
        plot.subplot(325), plot.plot(output[:params['label_sequence_length'], 0, 0], 'r', lw=2, label='X')
        plot.subplot(325), plot.plot(output[:params['label_sequence_length'], 0, 1], 'g', lw=2, label='Y')
        plot.subplot(325), plot.plot(output[:params['label_sequence_length'], 0, 2], 'b', lw=2, label='Z')
        plot.grid()
        plot.ylim([-1.1, 1.1])
        plot.title("Predicted XYZ - Source 0")
        plot.legend(loc='upper right') # Added Key

        # --- Predicted Source 1 ---
        plot.subplot(326), plot.plot(output[:params['label_sequence_length'], 1, 0], 'r', lw=2, label='X')
        plot.subplot(326), plot.plot(output[:params['label_sequence_length'], 1, 1], 'g', lw=2, label='Y')
        plot.subplot(326), plot.plot(output[:params['label_sequence_length'], 1, 2], 'b', lw=2, label='Z')
        plot.grid()
        plot.ylim([-1.1, 1.1])
        plot.title("Predicted XYZ - Source 1")
        plot.legend(loc='upper right') # Added Key

        # --- ADDED FORMATTING FIX ---
        plot.tight_layout()
        # --- END FIX ---

        if dump_figures:
            fig_name = '{}'.format(os.path.join(dump_folder, '{}.png'.format(os.path.basename(wav_file_path).split('.')[0])))
            print('saving figure : {}'.format(fig_name))
            plot.savefig(fig_name, dpi=100)
            plot.close()
        else:
            plot.show()

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)