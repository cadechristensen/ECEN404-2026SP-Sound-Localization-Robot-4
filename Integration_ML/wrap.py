import function_calls
import re
import serial
import time
from math import cos
from math import sin
from math import radians

ser = serial.Serial(
    port='/dev/serial0',
    baudrate=115200,
    timeout=1
)

def look_for_sound():
    #place holder for what will be the return of if there is a sound. if there is a sound also return the file
    time.sleep(0.1)
    pass

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
