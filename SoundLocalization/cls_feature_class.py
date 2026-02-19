import os
import numpy as np
import librosa

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

class FeatureClass:
    def __init__(self, params, is_eval=False):
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)
        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None
        self._is_eval = is_eval
        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)
        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)
        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._nb_mel_bins = params['nb_mel_bins']
        self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T
        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = 4
        self._use_hnet = params['use_hnet']
        self._nb_unique_classes = params['unique_classes']
        self._audio_max_len_samples = params['max_audio_len_s'] * self._fs 
        self._max_feat_frames = int(np.ceil(self._audio_max_len_samples / float(self._hop_len)))
        self._max_label_frames = int(np.ceil(self._audio_max_len_samples / float(self._label_hop_len)))
        
    def _process_loaded_audio(self, audio):
        if audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
             audio = audio.T
        elif audio.ndim == 1:
            audio = np.tile(audio[:, np.newaxis], (1, self._nb_channels))

        if audio.shape[1] < self._nb_channels:
            padding = np.zeros((audio.shape[0], self._nb_channels - audio.shape[1]))
            audio = np.hstack((audio, padding))
        elif audio.shape[1] > self._nb_channels:
            audio = audio[:, :self._nb_channels]

        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.zeros((self._audio_max_len_samples - audio.shape[0], audio.shape[1]))
            audio = np.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio

    def _load_audio(self, audio_path):
        audio, fs = librosa.load(audio_path, sr=self._fs, mono=False)
        return self._process_loaded_audio(audio), self._fs

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = np.zeros((self._max_feat_frames, nb_bins + 1, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra[:, :, ch_cnt] = stft_ch[:, :self._max_feat_frames].T
        return spectra

    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        W = linear_spectra[:, :, 0]
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        E = self._eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1))/3.0 )
        
        I_norm = I/E[:, :, np.newaxis]
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0,2,1)), self._mel_wts), (0,2,1))
        foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        return foa_iv

    def _get_gcc(self, linear_spectra):
        import math
        def nCr(n, r): return math.factorial(n) // math.factorial(r) // math.factorial(n-r)
        
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def get_normalized_wts_file(self):
        return os.path.join(self._feat_label_dir, '{}_wts'.format(self._dataset))

    def extract_features_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(audio_filename)
        return self._calculate_features(audio_in)

    def extract_features_from_memory(self, y, sr):
        if sr != self._fs:
            pass 
            
        audio_in = self._process_loaded_audio(y)
        return self._calculate_features(audio_in)

    def _calculate_features(self, audio_in):
        spect = self._spectrogram(audio_in)
        mel_spect = self._get_mel_spectrogram(spect)
        feat = None
        if self._dataset == 'foa':
            foa_iv = self._get_foa_intensity_vectors(spect)
            feat = np.concatenate((mel_spect, foa_iv), axis=-1)
        elif self._dataset == 'mic':
            gcc = self._get_gcc(spect)
            feat = np.concatenate((mel_spect, gcc), axis=-1)
        else:
            raise ValueError(f"ERROR: Unknown dataset format {self._dataset}")
        return feat