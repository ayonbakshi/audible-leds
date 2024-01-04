import socket

# Client setup
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
    host = '192.168.0.200'  # Change this to the server's IP address
    port = 8080

    client.connect((host, port))
    print(f"Connected to server at {host}:{port}")

    while True:
        data = client.recv(1024)
        if data == b'ping':
            client.send(b'pong')
