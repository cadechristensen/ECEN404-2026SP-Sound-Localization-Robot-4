"""
Audio Pipeline — real-time audio streaming and processing loop.

Contains the AudioPipeline class that manages PyAudio capture, the
processing thread (low-power detection → TTA confirmation → wake),
and the watchdog that restarts on failure.

Separated from detector.py so that offline tests can use BabyCryDetector
without pulling in PyAudio or threading dependencies.
"""

import sys
import time
import logging
import queue
import threading
import numpy as np

import pyaudio

from detector import BabyCryDetector, DetectionResult


class AudioPipeline:
    """
    Manages the real-time audio capture and processing loop.

    Takes a fully-initialised BabyCryDetector and wires it to a PyAudio
    stream, a processing thread, and a watchdog.
    """

    def __init__(self, detector: BabyCryDetector):
        self.detector = detector

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for continuous multi-channel audio capture."""
        if status:
            # logging is not re-entrant-safe inside a C-extension callback and can
            # deadlock on Pi when the logging handler's internal lock is already held
            # by the processing thread.  sys.stderr.write() is async-signal-safe.
            sys.stderr.write(f"[WARN] Audio callback status: {status}\n")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Reshape to preserve all channels (num_frames, num_channels)
        # This preserves phase relationships between channels for beamforming
        audio_data = audio_data.reshape(-1, self.detector.num_channels)

        # Add to queue for processing
        try:
            self.detector.audio_queue.put_nowait(audio_data)
        except queue.Full:
            sys.stderr.write("[WARN] Audio queue full, dropping frame\n")

        return (in_data, pyaudio.paContinue)

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def process_audio_stream(self):
        """Main processing loop for audio stream with optional temporal smoothing."""
        det = self.detector
        logging.info("Audio processing thread started")
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5
        _chunks_processed = 0
        _last_status_time = time.time()
        _STATUS_INTERVAL = 10.0  # Log audio activity every 10 seconds

        while det.is_running:
            try:
                # Get audio chunk from queue (timeout for responsiveness)
                audio_chunk = det.audio_queue.get(timeout=0.5)

                # Add to circular buffer
                det.audio_buffer.add(audio_chunk)
                _chunks_processed += 1

                # Periodic status log so the user knows audio is being captured
                now = time.time()
                if now - _last_status_time >= _STATUS_INTERVAL:
                    rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))
                    logging.info(
                        f"Audio active: {_chunks_processed} chunks processed, "
                        f"RMS={rms:.4f}"
                    )
                    _chunks_processed = 0
                    _last_status_time = now

                # Check cooldown
                if time.time() - det.last_detection_time < det.detection_cooldown:
                    continue

                # Don't fire detection until buffer has enough audio for SL
                if not det.audio_buffer.has_duration(det.context_duration):
                    continue

                if not hasattr(self, '_buffer_ready_logged'):
                    logging.info(
                        f"Audio buffer full ({det.context_duration:.0f}s) — ready to detect"
                    )
                    self._buffer_ready_logged = True

                consecutive_errors = 0  # Reset on successful queue read

                # Resample 48 kHz capture chunk to 16 kHz for gate checks.
                # The circular buffer keeps the original 48 kHz audio for DOAnet.
                bcd_chunk = det._resample_to_model_rate(audio_chunk)

                if det.low_power_mode:
                    # Low-power mode: gates-only per chunk, full pipeline on buffer.
                    # Run cheap energy + flatness gates on the best-RMS channel.
                    # No model inference per chunk — saves CPU and avoids the
                    # confidence mismatch between padded chunks and full windows.
                    if bcd_chunk.ndim > 1:
                        rms_per_ch = np.sqrt(np.mean(bcd_chunk ** 2, axis=0))
                        best_ch = int(np.argmax(rms_per_ch))
                        mono = bcd_chunk[:, best_ch]
                    else:
                        mono = bcd_chunk

                    gate_passed = (det._has_cry_band_energy(mono)
                                   and det._has_tonal_content(mono))

                    # Temporal smoothing on gate pass/fail
                    should_run_buffer_detection = False

                    if det.temporal_smoother is not None:
                        if not gate_passed:
                            logging.debug("Gate rejected — skipping temporal smoother update")
                            smoothed_result = None
                        else:
                            # Feed 1.0 for gate-pass to accumulate consecutive count
                            smoothed_result = det.temporal_smoother.update(1.0)
                            logging.debug(
                                f"Gate passed, consecutive={smoothed_result.consecutive_high_count}/"
                                f"{det.temporal_smoother.min_consecutive}"
                            )

                        if smoothed_result is not None and smoothed_result.should_alert:
                            should_run_buffer_detection = True
                            logging.info(
                                f"Temporal gate criteria met: "
                                f"{smoothed_result.consecutive_high_count} consecutive gate passes"
                            )
                    else:
                        should_run_buffer_detection = gate_passed

                    if should_run_buffer_detection:
                        # Run full multichannel pipeline on the buffer — same as
                        # prerecorded path. One pipeline, one confidence, one set
                        # of cry regions. No per-chunk model inference needed.
                        import torch

                        buffer_audio = det.audio_buffer.get_last_n_seconds(det.context_duration)
                        buffer_16k = det._resample_to_model_rate(buffer_audio)
                        sr = det.config.SAMPLE_RATE
                        threshold = det.detection_threshold

                        # Normalize to [-1, 1] for detection
                        buf_f32 = buffer_16k.astype(np.float32)
                        max_val = np.max(np.abs(buf_f32))
                        if max_val > 1.0:
                            buf_f32 = buf_f32 / max_val

                        audio_tensor = torch.from_numpy(buf_f32).float()

                        cry_regions = None
                        confidence = 0.0
                        try:
                            if buf_f32.ndim > 1 and buf_f32.shape[1] >= 2:
                                _, _, all_segs = det.audio_filter.isolate_baby_cry_multichannel(
                                    audio_tensor,
                                    cry_threshold=threshold,
                                    use_acoustic_features=False,
                                )
                            else:
                                mono_buf = audio_tensor[:, 0] if audio_tensor.ndim > 1 else audio_tensor
                                _, _, all_segs = det.audio_filter.isolate_baby_cry(
                                    mono_buf,
                                    cry_threshold=threshold,
                                    use_acoustic_features=False,
                                )

                            # Max confidence from all windows
                            if all_segs:
                                confidence = max(score for _, _, score in all_segs)

                            # Max-pool per sample then threshold
                            if all_segs:
                                total_samples = buf_f32.shape[0]
                                max_score = np.zeros(total_samples, dtype=np.float32)
                                for seg_start, seg_end, score in all_segs:
                                    s_idx = int(seg_start * sr)
                                    e_idx = min(int(seg_end * sr), total_samples)
                                    np.maximum(max_score[s_idx:e_idx], score, out=max_score[s_idx:e_idx])
                                cry_mask = max_score >= threshold
                                mask_int = cry_mask.view(np.int8)
                                edges = np.diff(mask_int, prepend=np.int8(0), append=np.int8(0))
                                starts = np.where(edges == 1)[0]
                                ends_arr = np.where(edges == -1)[0]
                                cry_regions = [(int(s) / sr, int(e) / sr)
                                               for s, e in zip(starts, ends_arr)]
                        except Exception as e:
                            logging.warning(f"Buffer detection failed: {e}")

                        if cry_regions:
                            logging.info(
                                f"Cry confirmed from buffer (confidence={confidence:.2%}, "
                                f"{len(cry_regions)} regions)"
                            )
                            detection = DetectionResult(
                                is_cry=True,
                                confidence=confidence,
                                timestamp=time.time(),
                                audio_buffer=buffer_audio,
                                filtered_audio=None,
                                cry_regions=cry_regions,
                            )
                            det.wake_robot(detection)
                            det.last_detection_time = time.time()
                        else:
                            logging.info(
                                f"Gates passed but no cry regions in buffer "
                                f"(max confidence={confidence:.2%}) — false positive"
                            )

                        if det.temporal_smoother is not None:
                            det.temporal_smoother.reset()

                else:
                    # Active mode: Robot is navigating
                    # Continue monitoring but don't wake again
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                consecutive_errors += 1
                logging.error(f"Error in audio processing ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}", exc_info=True)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logging.critical("Processing loop failed repeatedly — triggering safety alert")
                    det.processing_failed = True
                    break

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self, stream_audio: bool = True):
        """
        Start real-time detection.

        Args:
            stream_audio: Whether to start audio streaming (set False for testing)
        """
        det = self.detector
        logging.info("Starting real-time baby cry detector...")
        det.is_running = True

        # Model warmup FIRST: run dummy inferences to prime CPU kernel cache.
        # Must happen before the processing thread starts so the first real
        # predictions use warm caches and produce reliable latency/results.
        logging.info("Warming up model...")
        dummy_audio = np.zeros((det.chunk_size, det.num_channels), dtype=np.float32)
        import torch
        with torch.no_grad():
            for _ in range(3):
                det.detect_cry(dummy_audio)
        logging.info("Model warmup complete")

        # Start processing thread (after warmup so caches are primed)
        self._processing_thread = threading.Thread(
            target=self.process_audio_stream,
            daemon=True
        )
        self._processing_thread.start()

        # Start watchdog thread to monitor and restart processing on failure
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True
        )
        self._watchdog_thread.start()

        if stream_audio:
            # Initialize PyAudio
            self._pa = pyaudio.PyAudio()

            # List available devices
            if det.audio_device_index is None:
                logging.info("Available audio devices:")
                for i in range(self._pa.get_device_count()):
                    info = self._pa.get_device_info_by_index(i)
                    logging.info(f"  [{i}] {info['name']} (channels: {info['maxInputChannels']})")

            # Open audio stream
            device_index = det.audio_device_index

            try:
                self._stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=det.num_channels,
                    rate=det.capture_sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=det.chunk_size,
                    stream_callback=self._audio_callback
                )

                self._stream.start_stream()
                logging.info(
                    f"Audio stream started (device: {device_index}, channels: {det.num_channels}, "
                    f"capture: {det.capture_sample_rate}Hz, model: {det.config.SAMPLE_RATE}Hz)"
                )

            except Exception as e:
                logging.error(f"Error starting audio stream: {e}")
                logging.info("Detector running in test mode without audio input")

        logging.info("Real-time detector running in LOW-POWER MODE")

    def stop(self):
        """Stop real-time detection."""
        det = self.detector
        logging.info("Stopping real-time detector...")

        det.is_running = False

        # Join the processing thread *before* closing the stream so the thread
        # can finish draining audio_queue cleanly without racing against stream teardown.
        if hasattr(self, '_processing_thread'):
            self._processing_thread.join(timeout=2.0)

        if hasattr(self, '_stream'):
            self._stream.stop_stream()
            self._stream.close()

        if hasattr(self, '_pa'):
            self._pa.terminate()

        # Drain any remaining items from detection_queue
        while not det.detection_queue.empty():
            try:
                det.detection_queue.get_nowait()
            except queue.Empty:
                break

        logging.info("Real-time detector stopped")

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def _watchdog_loop(self):
        """
        Watchdog thread: monitors processing_failed flag and restarts the
        processing thread if it dies.  For a safety-critical baby monitor,
        silent failure is unacceptable.  Limits restarts to
        _max_restart_attempts to prevent infinite loops.
        """
        det = self.detector
        while det.is_running:
            time.sleep(2.0)  # Check every 2 seconds
            if not det.processing_failed:
                continue

            det._restart_attempts += 1
            if det._restart_attempts > det._max_restart_attempts:
                logging.critical(
                    "Processing thread failed %d times (max %d). "
                    "Giving up — manual intervention required.",
                    det._restart_attempts - 1, det._max_restart_attempts
                )
                break

            logging.critical(
                "WATCHDOG: processing_failed detected (attempt %d/%d). "
                "Restarting processing thread in 5 seconds...",
                det._restart_attempts, det._max_restart_attempts
            )
            time.sleep(5.0)

            # Reset flag and restart the processing thread
            det.processing_failed = False
            self._processing_thread = threading.Thread(
                target=self.process_audio_stream,
                daemon=True
            )
            self._processing_thread.start()
            logging.info("WATCHDOG: Processing thread restarted successfully")
