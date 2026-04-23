import argparse
import os
import numpy as np
import torch
import doanet_parameters
import doanet_model
import cls_feature_class

def get_predictions_for_file(audio_path, model_path, task_id):
    """
    Loads a trained DOAnet model and an audio file, and prints the predicted sound source locations.
    """
    model_path = "models/" + model_path
    # 1. Load the CORRECT parameters based on the task_id
    params = doanet_parameters.get_params(str(task_id)) 
    
    # 2. Dynamically determine model dimensions (no hardcoding)
    if params['dataset'] == 'foa':
        in_channels = 4 + 3  # 4 mel + 3 intensity vector
    elif params['dataset'] == 'mic':
        in_channels = 4 + 6  # 4 mel + 6 GCC-PHAT
    else:
        raise ValueError(f"Unknown dataset type: {params['dataset']}")
        
    output_classes = params['unique_classes'] * 3 # 3 coords (x,y,z) per source

    # ---- Set up device and model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_in = (params['batch_size'], in_channels, params['feature_sequence_length'], params['nb_mel_bins'])
    data_out = [params['batch_size'], params['label_sequence_length'], output_classes]
    
    model = doanet_model.CRNN(data_in, data_out, params).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---- Extract features ----
    # 3. Use the audio_path DIRECTLY from the command line
    feature_extractor = cls_feature_class.FeatureClass(params)
    features = feature_extractor.extract_features_for_file(audio_path)

    # ---- Reshape features ----
    time_steps = features.shape[0]
    mel_bins = params['nb_mel_bins']
    
    features_tensor = torch.from_numpy(features).view(time_steps, in_channels, mel_bins) 
    features_tensor = features_tensor.unsqueeze(0) 
    features_tensor = features_tensor.permute(0, 2, 1, 3).float().to(device) 

    # ---- Make prediction ----
    with torch.no_grad():
        if params['use_dmot_only']:
            output = model(features_tensor)
        else:
            output, _ = model(features_tensor)
            
    output = output.squeeze(0)

    # ---- Process output ----
    s1_locations = []
    s2_locations = []
    for frame_output in output:
        s1_coords = frame_output[0:3]
        s2_coords = frame_output[3:6]
        
        if torch.sqrt(torch.sum(s1_coords**2)) > 0.1:
            s1_locations.append(s1_coords)
        if torch.sqrt(torch.sum(s2_coords**2)) > 0.1:
            s2_locations.append(s2_coords)

    if s1_locations:
        s1_average = torch.mean(torch.stack(s1_locations), dim=0)
        print(f"\nSource 1 Average Location: X={s1_average[0]:.2f}, Y={s1_average[1]:.2f}, Z={s1_average[2]:.2f}")
    else:
        print("\nSource 1: Not detected.")

    if s2_locations:
        s2_average = torch.mean(torch.stack(s2_locations), dim=0)
        print(f"Source 2 Average Location: X={s2_average[0]:.2f}, Y={s2_average[1]:.2f}, Z={s2_average[2]:.2f}")
    else:
        print("Source 2: Not detected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for DOAnet.")
    # Add a new argument for the task ID
    parser.add_argument("task_id", type=int, help="The task ID (from parameters) the model was trained with.")
    parser.add_argument("audio_file", type=str, help="Path to the input WAV audio file.")
    parser.add_argument("model_file", type=str, help="Path to the trained .h5 model file.")
    args = parser.parse_args()
    
    # Pass all three arguments to the function
    get_predictions_for_file(args.audio_file, args.model_file, args.task_id)