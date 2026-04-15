#!/usr/bin/env python3
"""
Raspberry Pi setup and deployment script for baby cry detection.
Handles model optimization, audio configuration, and real-time detection.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

# This file lives in deployment/tools/ — insert the parent deployment/
# directory (for config_pi, etc.) and the project root (for src.*).
_DEPLOY_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def install_pi_dependencies():
    """Install Raspberry Pi specific dependencies."""
    print("Installing Pi dependencies...")

    # System dependencies for audio
    os.system("sudo apt-get update")
    os.system("sudo apt-get install -y portaudio19-dev python3-pyaudio")
    os.system("sudo apt-get install -y alsa-utils pulseaudio")

    # Python dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-pi.txt"])

    print("Pi dependencies installed!")

def optimize_model_for_pi(model_path: str, output_path: str):
    """Optimize model for Raspberry Pi deployment."""
    print(f"Optimizing model for Pi: {model_path}")

    from config_pi import ConfigPi
    from src.model import create_model

    config = ConfigPi()

    # Load model
    model = create_model(config)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Calibrated model: TemperatureScaledModel saved a second temperature
    # on top of BabyCryClassifier's own self.temperature.  Bake both into
    # model.temperature before quantizing so the calibrated scaling survives.
    if 'temperature' in checkpoint:
        model.temperature.data *= checkpoint['temperature']
        print(f"Calibrated temperature {checkpoint['temperature']:.4f} baked into model")

    model.eval()

    # Quantize for Pi
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    # Save optimized model
    torch.save({
        'model': quantized_model,
        'config': config.__dict__,
        'pi_optimized': True
    }, output_path)

    print(f"Pi-optimized model saved to: {output_path}")

def configure_audio():
    """Configure audio settings for optimal Pi performance."""
    print("Configuring audio for Pi...")

    # Set default audio device
    audio_config = """
# Pi Audio Configuration for Baby Cry Detection
pcm.!default {
    type hw
    card 0
}
ctl.!default {
    type hw
    card 0
}
"""

    with open("/tmp/asoundrc", "w") as f:
        f.write(audio_config)

    os.system("cp /tmp/asoundrc ~/.asoundrc")
    print("Audio configured!")

def test_pi_setup():
    """Test Pi setup and performance."""
    print("Testing Pi setup...")

    try:
        # Test PyAudio
        import pyaudio
        pa = pyaudio.PyAudio()
        info = pa.get_default_input_device_info()
        print(f"Audio input device: {info['name']}")
        pa.terminate()

        # Test AudioFilteringPipeline
        from audio_filtering import AudioFilteringPipeline
        from config_pi import ConfigPi

        config = ConfigPi()
        pipeline = AudioFilteringPipeline(config)
        print("AudioFilteringPipeline initialized successfully!")

        # Performance test with 4-channel 1-second audio
        print("Running performance test...")
        import time
        import numpy as np

        dummy_audio = np.random.randn(16000, config.PI_CHANNELS).astype(np.float32)
        start_time = time.time()

        pipeline.preprocess_audio(dummy_audio, apply_vad=False, apply_filtering=True)

        processing_time = time.time() - start_time
        real_time_factor = processing_time / 1.0  # 1 second of audio

        print(f"Processing time: {processing_time*1000:.1f}ms ({config.PI_CHANNELS}-channel)")
        print(f"Real-time factor: {real_time_factor:.2f}")

        if real_time_factor < 1.0:
            print("Pi setup ready for real-time processing!")
        else:
            print("Pi may struggle with real-time processing")

    except Exception as e:
        print(f"Test failed: {e}")

def create_pi_service():
    """Create systemd service for automatic startup."""
    print("Creating Pi service...")

    service_content = f"""
[Unit]
Description=Baby Cry Detection Service
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory={os.getcwd()}
ExecStart={sys.executable} deployment/realtime_baby_cry_detector.py
Restart=always
RestartSec=10
Environment=PYTHONPATH={os.getcwd()}

[Install]
WantedBy=multi-user.target
"""

    service_path = "/tmp/baby-cry-detection.service"
    with open(service_path, "w") as f:
        f.write(service_content)

    print(f"Service file created at: {service_path}")
    print("To install:")
    print(f"  sudo cp {service_path} /etc/systemd/system/")
    print("  sudo systemctl enable baby-cry-detection")
    print("  sudo systemctl start baby-cry-detection")

def main():
    """Main Pi setup function."""
    print("Raspberry Pi Baby Cry Detection Setup")
    print("=" * 50)

    import argparse
    parser = argparse.ArgumentParser(description='Setup baby cry detection on Pi')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install Pi dependencies')
    parser.add_argument('--optimize-model', type=str,
                       help='Path to model to optimize for Pi')
    parser.add_argument('--configure-audio', action='store_true',
                       help='Configure audio settings')
    parser.add_argument('--test', action='store_true',
                       help='Test Pi setup')
    parser.add_argument('--create-service', action='store_true',
                       help='Create systemd service')
    parser.add_argument('--all', action='store_true',
                       help='Run full setup')

    args = parser.parse_args()

    if args.all or args.install_deps:
        install_pi_dependencies()

    if args.all or args.configure_audio:
        configure_audio()

    if args.optimize_model:
        output_path = "model_pi_optimized.pth"
        optimize_model_for_pi(args.optimize_model, output_path)

    if args.all or args.test:
        test_pi_setup()

    if args.all or args.create_service:
        create_pi_service()

    if args.all:
        print("\nPi setup complete!")
        print("Run 'python deployment/realtime_baby_cry_detector.py --model <model_path>' to start detection")

if __name__ == "__main__":
    main()
