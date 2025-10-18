import serial
import time

class UnitV2SerialCommunication:
    def __init__(self, port='/dev/ttyAMA0', baudrate=115200, timeout=0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.is_connected = True
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            self.is_connected = False
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.is_connected = False
            print(f"Disconnected from {self.port}")

    def _check_connection(self):
        if not self.is_connected or not self.serial or not self.serial.is_open:
            raise ConnectionError("Not connected to serial port. Call connect() first.")

    def receive_data(self):
        self._check_connection()
        # read serial data until a newline character is received
        data = self.serial.readline().decode('utf-8')
        print(data)