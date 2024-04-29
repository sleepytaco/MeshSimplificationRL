import glob
import os
import subprocess

import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import requests
from settings import *


class MeshEnv(gym.Env):
    def __init__(self, mesh_files, final_face_count=100, training=True):
        super(MeshEnv, self).__init__()

        # subprocess.run(["/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/build-meshenv-Qt_6_2_4_for_macOS-Release/meshenv"])

        data = None
        try:
            # Send a GET request to the c++ mesh server
            response = requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/hello")
            data = response.json()
        except:
            assert data is not None, "Cannot connect to server :("

        # state space and action space for gym
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(STATE_SPACE_SIZE*3,), dtype=np.float32)
        self.action_space = spaces.Discrete(n=ACTION_SPACE_SIZE, start=0)

        self.mesh_files = []
        self.set_mesh_files(mesh_files)
        print(self.mesh_files[:10])

        # set env to training or testing mode, set mesh file for the env
        requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/update-env?action={'train' if training else 'test'}&faceCount={final_face_count}&")

    def set_training(self, isTraining):
        requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/update-env?action={'train' if isTraining else 'test'}&")

    def set_final_face_count(self, fc):
        requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/update-env?faceCount={fc}&")
    def set_mesh_files(self, mesh_files):
        self.mesh_files = mesh_files  # mesh_files can be a list or a string
        if type(mesh_files) is str:  # if a string is passed, it is referring to a folder path which holds meshes
            meshes_folder_path = mesh_files
            assert os.path.exists(meshes_folder_path), f"Mesh folder not found: {meshes_folder_path}"
            self.mesh_files = glob.glob(
                os.path.join(meshes_folder_path, '*.obj'))  # short cut to find all obj files in the folder path

            # recursively walk through all subfolders to look for .obj files
            self.mesh_files = []
            for root, dirs, files in os.walk(meshes_folder_path):
                for file in files:
                    if file.endswith(".obj") and file.count(".obj") == 1:
                        self.mesh_files.append(os.path.join(root, file))

    def set_rand_mesh(self):
        # pick a random mesh from the meshe_files and update the env with it
        mesh_file = random.choice(self.mesh_files)
        # update env with new mesh
        response = requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/update-env?meshFilePath={mesh_file}&")
        return response

    def reset(self, seed=None, options=None):
        # reset the environment to its initial state
        # response = requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/reset")

        # reset the enviroment to a random mesh
        response = self.set_rand_mesh()  # this updates the mesh env with a random mesh, and also resets the env state

        data = response.json()
        assert "state" in data, "'state' key not in the JSON returned by the server D:"
        state = np.array(data["state"], dtype=np.float32).flatten()
        info = {"message": data["message"]}
        return state, info

    def step(self, action):
        # take a step in the environment based on the given action
        # return the next observation, reward, done flag, and any additional information
        response = requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/step?action={action}&")
        data = response.json()
        for key in ["reward", "isTerminal", "state"]:
            assert key in data, f"'{key}' key not in the JSON returned by the server D:"

        state = np.array(data["state"], dtype=np.float32).flatten() #.view(STATE_SPACE_SIZE * 3, -1)
        reward = data["reward"]
        done = truncated = data["isTerminal"]
        info = {"message": data["message"]}
        return state, reward, done, truncated, info

    def render(self, mode='human'):
        # render the current state of the meshenv (optional)
        pass

    def close(self):
        # clean up resources (optional)
        # save mesh state, shutdown c++ server
        requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/bye")
