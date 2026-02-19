def get_params(argv='1'):
    params = dict(
        quick_test=False, 
        dataset_dir='DCASE2020_SELD_dataset/', 
        feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/', 
        model_dir='models/',   
        dcase_dir='results/', 
        mode='dev',         
        dataset='mic',      
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,
        use_hnet=True,
        label_sequence_length=50,    
        batch_size=64,             
        dropout_rate=0.,             
        nb_cnn2d_filt=128,          
        f_pool_size=[2, 2, 2],      
        nb_rnn_layers=2,
        rnn_size=128,        
        self_attn=False,
        nb_heads=4,
        nb_fnn_layers=2,
        fnn_size=128,             
        nb_fnn_act_layers=2,
        fnn_act_size=128,             
        nb_epochs=200,              
        lr=1e-3,
        dMOTA_wt = 1,
        dMOTP_wt = 50,
        IDS_wt = 1,
        branch_weights=[1, 10.],
        use_dmot_only=False,
    )
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")
        params['self_attn']= True
        params['lr'] = 1e-4 
        params['nb_epochs'] = 300   
        params['nb_rnn_layers']=3
        params['rnn_size']=256

    elif argv == '2': #Second attempt
        print("USING SECOND PARAMETERS\n")
        params['self_attn']= False
        params['lr'] = 1e-4
        #params['rnn_size'] = 256
        #params['nb_cnn2d_filt'] = 256
    elif argv == '3': #Third attempt
        print("USING THIRD PARAMETERS\n")
        params['self_attn']= True
        params['lr'] = 1e-4
        params['nb_epochs'] = 300
    elif argv == '4': #Fourth attempt
        print("USING FOURTH PARAMETERS\n")
        params['self_attn']= True
        params['lr'] = 1e-4
        params['nb_epochs'] = 300   
        params['rnn_size']=256
    elif argv == '5': #Fifth attempt
        print("USING FIFTH PARAMETERS\n")
        params['self_attn']= True
        params['lr'] = 1e-4
        params['nb_epochs'] = 300   
        params['nb_rnn_layers']=3
        
    elif argv == '6': #sixth attempt
        print("USING SIXTH PARAMETERS\n")
        params['self_attn']= True
        params['lr'] = 1e-3 
        params['nb_epochs'] = 300   
        params['nb_rnn_layers']=3
        params['rnn_size']=256
        
    elif argv == '7': #7th attempt
        print("USING SEVENTH PARAMETERS\n")
        params['self_attn']= True
        params['lr'] = 1e-4 
        params['nb_epochs'] = 300   
        params['nb_rnn_layers']=3
        params['rnn_size']=256

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]    
    params['patience'] = int(params['nb_epochs'])     
    params['unique_classes'] = 2 
    return params