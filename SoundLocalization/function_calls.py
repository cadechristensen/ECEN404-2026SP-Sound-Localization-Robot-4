import logging
import os
import sys
import warnings
import numpy as np
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

logger = logging.getLogger(__name__)


class Infer:
    #! Channel-to-angle mapping (from doanet_parameters)
    CHANNEL_ANGLES = {0: 135.0, 1: 315.0, 2: 45.0, 3: 225.0}

    #! RMS ratio threshold — if loudest / second < this, sound is between them
    #! e.g. 1.3 means the loudest must be 30% louder to be considered dominant
    RMS_RATIO_THRESHOLD = 1.3

    #! Degree-specific model files
    _MODEL_FILES = {
        # 0:   "SL_0deg_model.h5",
        0: "New_Test_Model.h5",
        45: "New_Test_Model.h5",
        90: "New_Test_Model.h5",
        # 135: "SL_135deg_and_180deg_model.h5",
        135: "315test.h5",
        180: "SL_135deg_and_180deg_model.h5",
        225: "New_Test_Model.h5",
        # 270: "SL_225deg_and_315deg_model.h5",
        270: "315test.h5",
        # 315: "SL_225deg_and_315deg_model.h5",
        315: "315test.h5",
    }

    #! Adjacent mic pairs → midpoint angle (for "between" detection)
    _BETWEEN_MAP = {
        frozenset([45, 135]): 90,
        frozenset([135, 225]): 180,
        frozenset([225, 315]): 270,
        frozenset([315, 45]): 0,
    }

    def __init__(self, task_id="6", models_dir=None, single_model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = doanet_parameters.get_params(task_id)
        self.params.update(
            {
                "nb_cnn2d_filt": 128,
                "rnn_size": 256,
                "self_attn": True,
                "unique_classes": 2,
            }
        )
        nb_ch = 10
        self.seld_ch = nb_ch
        self._sl_dir = os.path.dirname(os.path.abspath(__file__))

        #! Resolve models directory — all inference artifacts live here
        if models_dir is None:
            models_dir = os.path.join(self._sl_dir, "models")
        self._models_dir = os.path.abspath(models_dir)

        data_in = (
            self.params["batch_size"],
            nb_ch,
            self.params["feature_sequence_length"],
            self.params["nb_mel_bins"],
        )
        data_out = [
            self.params["batch_size"],
            self.params["label_sequence_length"],
            self.params["unique_classes"] * 3,
        ]

        if single_model:
            #! Single model mode — one checkpoint for all angles
            path = os.path.join(self._models_dir, single_model)
            if not os.path.exists(path):
                raise FileNotFoundError(f"SL model not found: {path}")
            model = doanet_model.CRNN(data_in, data_out, self.params).to(self.device)
            model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            model.eval()
            self.models = {deg: model for deg in self._MODEL_FILES}
            print(f"Loaded single SL model ({single_model}) for all angular regions")
        else:
            #! Load all degree-specific models (deduplicate shared files)
            unique_models = {}
            for fname in set(self._MODEL_FILES.values()):
                path = os.path.join(self._models_dir, fname)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"SL model not found: {path}")
                model = doanet_model.CRNN(data_in, data_out, self.params).to(
                    self.device
                )
                model.load_state_dict(
                    torch.load(path, map_location=self.device, weights_only=True)
                )
                model.eval()
                unique_models[fname] = model

            self.models = {
                deg: unique_models[fname] for deg, fname in self._MODEL_FILES.items()
            }
            print(
                f"Loaded {len(unique_models)} SL models for {len(self.models)} angular regions"
            )

        #! Load normalization scaler
        scaler_path = os.path.join(self._models_dir, "mic_wts")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Normalization weights not found: {scaler_path}")
        self.scaler_seld = joblib.load(scaler_path)

        #! Distance prediction is handled externally by
        #! Pi_Integration/record_samtry.py::predict_ml_distance. This class
        #! only does direction-of-arrival; the caller computes distance from
        #! the raw 48kHz audio.

        print("System Ready.")

    # ------------------------------------------------------------------
    #! Model selection based on channel RMS
    # ------------------------------------------------------------------
    def _compute_channel_rms(self, y_multichannel):
        """Compute cry-band RMS (300-3000Hz) per channel.

        Args:
            y_multichannel: shape (num_channels, num_samples) from librosa.

        Returns:
            List of (channel_index, rms) sorted by RMS descending.
        """
        from scipy.signal import butter, sosfilt

        if y_multichannel.ndim == 1:
            return [(0, float(np.sqrt(np.mean(y_multichannel**2))))]

        sr = 48000
        sos = butter(4, [300, 3000], btype="bandpass", fs=sr, output="sos")

        channel_rms = []
        for ch in range(y_multichannel.shape[0]):
            filtered = sosfilt(sos, y_multichannel[ch])
            rms = float(np.sqrt(np.mean(filtered**2)))
            channel_rms.append((ch, rms))

        channel_rms.sort(key=lambda x: x[1], reverse=True)
        return channel_rms

    def _select_model(self, y_multichannel):
        """Select the best degree-specific model based on channel RMS.

        Returns:
            (model, selected_degree, info_string)
        """
        channel_rms = self._compute_channel_rms(y_multichannel)

        top1_ch, top1_rms = channel_rms[0]
        top2_ch, top2_rms = channel_rms[1] if len(channel_rms) > 1 else (None, 0.0)

        top1_angle = int(self.CHANNEL_ANGLES[top1_ch])
        top2_angle = int(self.CHANNEL_ANGLES[top2_ch]) if top2_ch is not None else None

        rms_ratio = top1_rms / top2_rms if top2_rms > 0 else 999.0

        if top2_angle is not None and rms_ratio < self.RMS_RATIO_THRESHOLD:
            # Similar RMS — sound is between the two mics
            pair = frozenset([top1_angle, top2_angle])
            if pair in self._BETWEEN_MAP:
                selected_deg = self._BETWEEN_MAP[pair]
                info = (
                    f"Between {top1_angle}° and {top2_angle}° "
                    f"(ratio={rms_ratio:.2f} < {self.RMS_RATIO_THRESHOLD}) "
                    f"-> {selected_deg}° model"
                )
            else:
                # Non-adjacent pair (opposite mics) — fall back to dominant
                selected_deg = top1_angle
                info = (
                    f"Non-adjacent {top1_angle}°+{top2_angle}° "
                    f"(ratio={rms_ratio:.2f}) -> dominant {selected_deg}° model"
                )
        else:
            # One channel clearly louder — sound is at that mic
            selected_deg = top1_angle
            info = (
                f"Dominant {top1_angle}° "
                f"(ratio={rms_ratio:.2f} >= {self.RMS_RATIO_THRESHOLD}) "
                f"-> {selected_deg}° model"
            )

        model = self.models[selected_deg]
        rms_summary = ", ".join(
            f"Ch{ch}({self.CHANNEL_ANGLES[ch]:.0f}°)={rms:.4f}"
            for ch, rms in channel_rms
        )
        logger.debug(f"Channel RMS: {rms_summary}")
        logger.debug(f"Model selection: {info}")
        return model, selected_deg, info

    # ------------------------------------------------------------------
    #! Inference
    # ------------------------------------------------------------------
    def process_file(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return f"Error: File {filepath} not found."

        res_str = []
        try:
            y_raw, sr_raw = librosa.load(filepath, sr=48000, mono=False)

            # Select best model based on channel RMS
            model, selected_deg, sel_info = self._select_model(y_raw)

            if self.params["fs"] != 48000:
                y_seld = librosa.resample(
                    y_raw, orig_sr=48000, target_sr=self.params["fs"], res_type="scipy"
                )
            else:
                y_seld = y_raw

            feat_extractor = cls_feature_class.FeatureClass(self.params)
            features_SELD = feat_extractor.extract_features_for_file(filepath)
            features_SELD = self.scaler_seld.transform(features_SELD)

            feat_seq_len = self.params["feature_sequence_length"]
            nb_feat_frames = features_SELD.shape[0]
            batch_size_feat = int(np.ceil(nb_feat_frames / float(feat_seq_len)))
            feat_pad_len = batch_size_feat * feat_seq_len - nb_feat_frames
            if feat_pad_len > 0:
                features_SELD = np.pad(
                    features_SELD,
                    ((0, feat_pad_len), (0, 0)),
                    "constant",
                    constant_values=1e-6,
                )

            features_SELD = features_SELD.reshape(
                (
                    batch_size_feat,
                    feat_seq_len,
                    self.seld_ch,
                    self.params["nb_mel_bins"],
                )
            )
            features_SELD = np.transpose(features_SELD, (0, 2, 1, 3))
            data_SELD = torch.tensor(features_SELD).to(self.device).float()

            output, activity_out = model(data_SELD)

            max_nb_doas = output.shape[2] // 3
            output = output.view(
                output.shape[0], output.shape[1], 3, max_nb_doas
            ).transpose(-1, -2)
            output = output.view(-1, output.shape[-2], output.shape[-1])
            activity_out = activity_out.view(-1, activity_out.shape[-1])

            output = output.cpu().detach().numpy()
            sigmoid_scores = torch.sigmoid(activity_out).cpu().detach().numpy()

            real_samples = y_seld.shape[1] if y_seld.ndim > 1 else y_seld.shape[0]
            hop_len_samples = self.params["hop_len_s"] * self.params["fs"]
            hop_ratio = self.params["label_hop_len_s"] / self.params["hop_len_s"]
            nb_feat_frames_real = int(np.ceil(real_samples / float(hop_len_samples)))
            nb_label_frames = int(np.ceil(nb_feat_frames_real / hop_ratio))

            output_real = output[:nb_label_frames]
            sigmoid_scores_real = sigmoid_scores[:nb_label_frames]
            activity_real = sigmoid_scores_real > 0.1

            # Collect all detected sources
            sources = []
            for i in range(2):
                mask = activity_real[:, i]
                if np.any(mask):
                    x, y_coords = output_real[mask, i, 0], output_real[mask, i, 1]
                    deg = np.degrees(np.arctan2(np.mean(y_coords), np.mean(x))) % 360
                    conf = np.mean(sigmoid_scores_real[mask, i])
                    sources.append((deg, conf))

            # Pick the source closest to the estimated direction
            # (selected_deg = midpoint for "between" cases, mic angle for dominant)
            if sources:

                def angular_dist(a, b):
                    return min((a - b) % 360, (b - a) % 360)

                # --- NEW OVERRIDE FOR 0 DEGREES ---
                if selected_deg == 0:
                    # Sort by whichever is closer: 0 degrees OR 90 degrees
                    sources.sort(
                        key=lambda s: min(angular_dist(s[0], 0), angular_dist(s[0], 90))
                    )
                    logger.debug(
                        f"Source selection (0-deg override): picked {sources[0][0]:.1f}° "
                        f"(closest to 0° or 90°)"
                    )
                else:
                    # Standard logic for all other degrees
                    sources.sort(key=lambda s: angular_dist(s[0], selected_deg))
                    logger.debug(
                        f"Source selection: picked {sources[0][0]:.1f}° "
                        f"(closest to estimated {selected_deg}°)"
                    )

            for i, (deg, conf) in enumerate(sources):
                res_str.append(f"Source {i}: {deg:.1f}° (Loudness: {conf:.2f})")

        except Exception as e:
            res_str.append(f"Inference Error: {e}")

        return " | ".join(res_str) if res_str else "No active sources detected."


def record_audio(
    filename: str = "live_input.wav",
    duration: float = 10.0,
    sample_rate: int = 48000,
    channels: int = 4,
    chunk_size: int = 1024,
) -> str:
    audio = pyaudio.PyAudio()
    stream = None
    try:
        device_index = None
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info["maxInputChannels"] >= channels:
                device_index = i
                break
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
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
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
        audio_data = audio_data.reshape(-1, channels).flatten()

        with wave.open(filename, "wb") as wf:
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
