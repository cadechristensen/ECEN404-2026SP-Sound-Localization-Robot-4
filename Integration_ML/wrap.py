import function_calls
import re
import serial
import time
import argparse
import logging
from math import cos
from math import sin
from math import radians

ser = serial.Serial(
    port='/dev/serial0',
    baudrate=115200,
    timeout=1
)

def look_for_sound():
    """Main integrated baby monitor - low-power autonomous mode."""
    parser = argparse.ArgumentParser(description='Integrated Baby Monitor System')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained baby cry detection model')
    parser.add_argument('--device-index', type=int, default=None,
                       help='Audio device index for microphone')
    parser.add_argument('--channels', type=int, default=4,
                       help='Number of microphone channels (default: 4)')
    parser.add_argument('--no-uart', action='store_true',
                       help='Disable UART communication (for testing)')
    parser.add_argument('--no-email', action='store_true',
                       help='Disable email notifications (for testing)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 70)
    print("AUTONOMOUS BABY MONITOR ROBOT")
    print("=" * 70)
    print("\nSystem Architecture:")
    print("  [LOW-POWER MODE] -> Quick 1-sec detection")
    print("       |")
    print("       v (3 consecutive positives)")
    print("  [WAKE UP] -> Capture 3s context + TTA confirmation")
    print("       |")
    print("       v (>85% confidence)")
    print("  [ACTIVE MODE]")
    print("    1. Record 5s for localization (48kHz)")
    print("    2. Run DOAnet (direction + distance)")
    print("    3. Send email notification")
    print("    4. Navigate robot via ESP32")
    print("       |")
    print("       v")
    print("  [RETURN TO LOW-POWER MODE]")
    print("\n" + "=" * 70)

    # Initialize UART
    if not args.no_uart:
        if not init_uart():
            logging.warning("UART init failed, running without robot control")
    else:
        logging.info("UART disabled (test mode)")

    # Create monitor (will set callback after handler is created)
    monitor = LowPowerBabyMonitor(
        model_path=args.model,
        device_index=args.device_index,
        num_channels=args.channels,
        on_cry_confirmed=None  # Set below
    )

    # Create response handler
    handler = CryResponseHandler(
        monitor=monitor,
        enable_email=not args.no_email
    )

    # Connect callback
    monitor.on_cry_confirmed = handler.handle_cry_detected

    # Clear UART buffer
    if ser is not None:
        time.sleep(1)
        ser.reset_input_buffer()

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Channels: {args.channels}")
    print(f"  UART: {'Enabled' if ser is not None else 'Disabled'}")
    print(f"  Email: {'Enabled' if not args.no_email else 'Disabled'}")
    print(f"\nEntering LOW-POWER listening mode...")
    print("Press Ctrl+C to stop\n")

    try:
        # Run low-power listening loop
        monitor.run_low_power_loop()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        monitor.stop()

    finally:
        if ser is not None:
            ser.close()
        print("System stopped.")
    time.sleep(0.1)
    return

def process_sound_file(sound_file):
    engine = function_calls.Infer()
    result_string = engine.process_file(sound_file)
    dist = re.search(r"Distance: (\d+\.?\d*) ft", result_string)
    distance = float(dist.group(1))
    deg = re.search(r"Source \d+: (\d+\.?\d*)°", result_string)
    degrees = float(deg.group(1))
    rads=radians(degrees)
    x=distance*cos(rads)
    y=distance*sin(rads)
    return (x, y, distance) 

def drive_robot(coordinates):
    # placeholder for robot algortihm that will navigate and then send back to start
    x, y, distance = coordinates
    
    cmd = f"NAV x={x:.3f} y={y:.3f} d={distance:.3f}\n"
    print(f"Sending → ESP32: {cmd.strip()}")

    ser.write(cmd.encode())
    time.sleep(0.3)
    '''Obstacle avoidance algorithm is now running on ESP32
    until obstacle is avoided to relisten'''
    #Main loop handles relistening/obstacle communication
    return



def main():
    print("Listening for baby crying...")
    time.sleep(1)
    ser.reset_input_buffer()
    
    while True:
        sound_file = None        
        while True:
            is_sound_detected, sound_file = look_for_sound()
            if is_sound_detected:
                break 
            time.sleep(0.1)

        print("Sound detected!")
        xyz = process_sound_file(sound_file) # returns x,y, distance in a tuple

        if xyz is None:
            print("Invalid sound data")
            continue
            
        drive_robot(xyz)    
        #Wait for ESP32 to finish navigating
        print("Waiting for EP32...")
        
        while True:
            msg = ser.readline().decode().strip()
            if not msg:
                continue
                
            print("RX:", msg)
            
            if msg == "RELISTEN":
                print("Obstacle avoided. Relistening...")
                break
            elif msg == "OBSTACLE":
                print("ESP32 hit obstacle. Navigating around...")   
            elif msg == "READY":
                print("ESP32 ready")
        time.sleep(0.5)

        #    back to the start to listen again

   

if __name__ == "__main__":
    main()