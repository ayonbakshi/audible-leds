import socket
import time

import numpy as np

from audio import audio_spectrum

def interleave(*arrs):
    c = np.empty(len(arrs[0])*len(arrs), dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        c[i::len(arrs)] = arr
    return c

def bins_to_red(bins):
    ones = (254 * bins**2).astype(np.uint8)
    zeros = np.zeros(len(bins), dtype=np.uint8)
    return interleave(ones, zeros, ones)

def send_led_bytes(arr):
    return arr.tobytes() + b'\xff'

SERVER_IP = '192.168.4.1'  # Replace with ESP32's IP
SERVER_PORT = 12345

FPS = 60

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, SERVER_PORT))
    print("Connected to ESP32 server")

    start_millis = None
    last = time.time()
    for bins in audio_spectrum():
        print(max(bins))
        b = send_led_bytes(bins_to_red(bins))
        client_socket.send(b)
        _last = time.time()
        time.sleep(max(0, 1/FPS-(_last-last)))
        last = _last

        # client_socket.send(b'0'*300 + b'\xff')

        # start_time = time.time()
        # client_socket.send(b'0' * 100 + b'\xff')
        # response = client_socket.recv(1024)
        # end_time = time.time()
        
        # if start_millis == None:
        #     start_millis = end_time * 1000 - int(response)
        # my_millis = end_time * 1000 - start_millis
        # round_trip_time = end_time - start_time
        # print(f"Received pong. Round-trip time: {round_trip_time:.4f} seconds")
        # print(response, round(my_millis))

        time.sleep(0.01)  # Wait before sending the next ping

    client_socket.close()

if __name__ == "__main__":
    main()
