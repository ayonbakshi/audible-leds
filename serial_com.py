import serial
import time
import socket
import collections
import numpy as np

from audio import audio_spectrum

import matplotlib.pyplot as plt

# Replace 'COMx' with your ESP32's serial port name (e.g., '/dev/cu.usbserial-XXXXX' on macOS)
serial_port = '/dev/cu.usbserial-120'  # Example port name

UDP_IP = "192.168.0.154"  # Replace with your ESP32's IP address
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP

# Open serial connection (make sure to adjust baudrate as per your ESP32 settings)
ser = serial.Serial(serial_port, baudrate=460800, timeout=0.1)

def interleave(*arrs):
    c = np.empty(len(arrs[0])*len(arrs), dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        c[i::len(arrs)] = arr
    return c

def bins_to_red(bins):
    ones = (254 * bins**2).astype(np.uint8)
    zeros = np.zeros(len(bins), dtype=np.uint8)
    return interleave(ones, zeros, ones)

def bins_to_cm(cmap, bins):
    m = plt.get_cmap(cmap)
    cols = m(bins)
    cols = cols[:,:-1]

    factor = np.clip(np.repeat(bins, 3, axis=-1).reshape(-1, 3), 0.2, 1)
    cols = factor * cols

    cols = (254 * cols**1.0).astype(np.uint8)
    return cols.flatten()

def send_led_bytes(arr):
    return arr.tobytes()# + b'\xff'

num_leds = 100

if ser.is_open:
    print(f"Serial port {serial_port} opened")

    try:
       
        # while True:
        #     b = ser.read_all().decode('ascii', errors='replace')
        #     print(b, end='')
        #     if 'Listening' in b:
        #         start_local = time.time() * 1000
        #         l = b.strip()
        #         l = l[l.rfind('\n')+1:]
        #         start_device = int(l[:l.find(" ")])
        #         break

        messages = collections.deque([])
        last_time = 0
        for bins in audio_spectrum():
        # while True:
        # for i in range(256):
            # Send data to ESP32 (replace 'Hello!' with your desired data)
            # ser.write(b'0'*300 + b'\xff')
            # bins = i * np.ones(num_leds) / 255
            bin_bytes = send_led_bytes(bins_to_cm('inferno', bins))
            # ser.write(b)
            # print("sent packet")
            sock.sendto(bin_bytes, (UDP_IP, UDP_PORT))
            b = ser.read_all().decode('ascii', errors='replace')
            print(b, end='')

            # if 'at' in b:
            #     device_time = int(b[b.find('at')+2:b.find(':')])
            #     print(f'diff {device_time - last_time}'); last_time = device_time
                # local_time = device_time - start_device + start_local
                # acked_msg_t = messages.popleft()
                # print(f'local time diff {local_time - acked_msg_t}')

            time.sleep(0.01)  # Adjust delay as needed
    except KeyboardInterrupt:
        ser.close()
        print("Serial port closed")
else:
    print(f"Failed to open serial port {serial_port}")