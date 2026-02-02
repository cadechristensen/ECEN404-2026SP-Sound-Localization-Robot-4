import function_calls
import re
from math import cos
from math import sin
from math import radians


def look_for_sound():
    #place holder for what will be the return of if there is a sound. if there is a sound also return the file
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
    pass



def main():
    while True:
        sound_file = None        
        while True:
            is_sound_detected, sound_file = look_for_sound()
            if is_sound_detected:
                break 
            
        xyz = process_sound_file(sound_file) # returns x,y, distance in a tuple
        
        drive_robot(xyz)
        
        #    back to the start to listen again


    
def main():
    filename = function_calls.record_audio()
    if filename:
        engine = function_calls.AudioInferenceEngine()
        result_string = engine.process_file(filename)
        degrees = None
        distance = None
        dist = re.search(r"Distance: (\d+\.?\d*) ft", result_string)
        distance = float(dist.group(1))
        deg = re.search(r"Source \d+: (\d+\.?\d*)°", result_string)
        degrees = float(deg.group(1))
        coordinates = (distance, degrees)


if __name__ == "__main__":
    main()