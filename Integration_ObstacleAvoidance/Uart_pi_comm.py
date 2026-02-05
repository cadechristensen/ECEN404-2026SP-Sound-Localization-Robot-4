"This will be programmed on Raspberry Pi to communicate to ESP32 through UART"
import serial
import time

class UARTComm:
    def __init__(self, port="/dev/serial0", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        print(f"UART initialized on {port}")
    
    def wait_for_listen(self, timeout=30):
        """Wait for ESP32 to send LISTEN command."""
        print("Waiting for ESP32 to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.ser.in_waiting:
                msg = self.ser.readline().decode().strip()
                print(f"Received: {msg}")
                
                if msg == "LISTEN":
                    return True
            
            time.sleep(0.1)
        
        print("WARNING: Timeout waiting for LISTEN")
        return False
    
    def send_command(self, command):
        """
        Send movement command to ESP32.
        Format: "FORWARD 0.75\n" or "LEFT 0.5\n" etc.
        """
        self.ser.write(command.encode())
        print(f"Sent: {command.strip()}")
    
    def wait_for_done(self, timeout=60):
        """Wait for ESP32 to finish movement."""
        print("Waiting for ESP32 to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.ser.in_waiting:
                msg = self.ser.readline().decode().strip()
                print(f"Received: {msg}")
                
                if msg == "DONE":
                    return True
            
            time.sleep(0.1)
        
        print("WARNING: Timeout waiting for DONE")
        return False
    
    def close(self):
        self.ser.close()
