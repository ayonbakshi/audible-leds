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
    red = (254 * bins).astype(np.uint8)
    zeros = np.zeros(len(bins), dtype=np.uint8)
    return interleave(red, zeros, zeros)

def send_led_bytes(arr):
    return arr.tobytes() + b'\xff'

def main():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        host = '192.168.0.200'  # Change this to your server's IP address if needed
        port = 8080

        server.bind((host, port))
        server.listen(0)

        print(f"Server listening on {host}:{port}")

        # Accept incoming connection
        while True:
            client_socket, client_address = server.accept()
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
            print(f"Connection from {client_address} established.")

            # Ping the client
            on = 0
            leds = 100


            for bins in audio_spectrum():
                try:
                    print(max(bins))
                    b = send_led_bytes(bins_to_red(bins))
                    client_socket.send(b)

                    # cols = np.array([1, 0, 0], dtype=np.uint8)
                    # b = send_led_bytes(on * cols)
                    # # print(f'sending {b}')
                    # if on == 0: print("start")
                    # client_socket.send(b)
                    # on += 1
                    # if on == 255: on = 0

                    # start_time = time.time()
                    # client_socket.send(b'ping')
                    # data = client_socket.recv(1024)
                    # end_time = time.time()
                    # latency = (end_time - start_time) * 1000  # Calculate latency in milliseconds
                    # print(f"Got message {data}. Ping latency: {latency:.2f} ms")
                    time.sleep(1/60)  # Ping every 1 second
                except BrokenPipeError:
                    print("Client died. Restarting.")
                    break

if __name__ == "__main__":
    main()