"""
This module integrates the baby cry detection, sound localization, and robot control into a cohesive system. It listens for baby cries, processes the audio to determine direction and distance, and sends navigation commands to the ESP32-controlled robot. The system is designed to operate in a low-power mode until a cry is detected, at which point it wakes up to perform more intensive processing and control tasks.
"""
import function_calls
import function_calls2
import re
import serial
import time
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
    """
    Record audio from the mic array, run baby cry detection with noise
    filtering, and return (detected: bool, filtered_file_path: str).

    Low-power mode: a cheap detect_cry() call runs first (no TTA).
    The expensive confirm_and_filter() (with TTA) only fires when the
    quick screen detects a potential cry.
    """
    # Record and resample (48 kHz -> 16 kHz)
    audio_48k, audio_16k = function_calls2.record_and_resample()
    if audio_16k is None:
        logging.warning("Recording failed")
        return (False, "")

    # Low-power quick screening - no TTA
    is_cry, confidence = function_calls2.detect_cry(audio_16k)
    logging.info(f"Quick screen: is_cry={is_cry}, confidence={confidence:.2%}")
    if not is_cry:
        return (False, "")

    # Confirm with TTA + isolate cry segments
    result = function_calls2.confirm_and_filter(audio_16k)
    logging.info(f"Confirmation: is_cry={result.is_cry}, confidence={result.confidence:.2%}")
    if not result.is_cry or result.confidence < 0.85:
        return (False, "")

    # Save filtered audio at 48 kHz for DOAnet localization
    filtered_path = function_calls2.save_filtered_audio(result.filtered_audio)
    if filtered_path is None:
        # Fallback: save original 48 kHz recording
        filtered_path = "filtered_cry.wav"
        import soundfile as sf
        sf.write(filtered_path, audio_48k, function_calls2.MIC_SAMPLE_RATE)

    logging.info(f"Filtered baby cry saved to {filtered_path}")
    return (True, filtered_path)

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
    ans=False
    x, y, distance = coordinates
    cmd = f"NAV x={x:.3f} y={y:.3f} d={distance:.3f}\n"
    print(f"Sending → ESP32: {cmd.strip()}")

    ser.write(cmd.encode())
    time.sleep(0.3)
    '''Obstacle avoidance algorithm is now running on ESP32
    until obstacle is avoided to relisten'''
    while True:
        msg = ser.readline().decode().strip()
        if not msg:
            continue
                
        print("RX:", msg)
            
        if msg == "RELISTEN":
            print("Obstacle avoided. Relistening...")
            ans=False
            break
        elif msg == "READY":
            ans=True
            print("ESP32 ready")
    #Main loop handles relistening/obstacle communication
    return ans

def sendtouser():
    pass




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
            
        decision=drive_robot(xyz)    
        #Wait for ESP32 to finish navigating
        if decision==True:
            sendtouser()
            break
        else:
            time.sleep(0.1)
            continue



        #    back to the start to listen again

   

if __name__ == "__main__":
    main()