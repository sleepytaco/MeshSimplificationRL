# This file includes sample code to interact with the MeshEnvCPP server

import requests
import time

MESHENV_SERVER_HOST = '127.0.0.1'  # Server IP address
MESHENV_SERVER_PORT = 12345        # Server port
ENDPOINT = '/reset'

start_time = time.time()
for _ in range(1):
    for _ in range(400):
        try:
            # Send a GET request to the meshenv server
            response = requests.get(f"http://{MESHENV_SERVER_HOST}:{MESHENV_SERVER_PORT}{ENDPOINT}")
            data = response.json()
            # print("Received data:", data)
        except:
            print("Cannot connect to server :(")
end_time = time.time()
elapsed_time = end_time - start_time
print("Time taken:", elapsed_time, "seconds")
