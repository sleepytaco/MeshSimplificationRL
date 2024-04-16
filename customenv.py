from gym import spaces, Env
import numpy as np
import requests
from settings import *
from torch import tensor, float32


class MeshEnv(Env):
    def __init__(self, training=True):
        super(MeshEnv, self).__init__()

        data = None
        try:
            # Send a GET request to the c++ mesh server
            response = requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/hello")
            data = response.json()
        except:
            assert data is not None, "Cannot connect to server :("

        # state space and action space for gym
        self.observation_space = spaces.Box(low=-10000000, high=1000000, shape=(STATE_SPACE_SIZE*3,), dtype=np.float32)
        self.action_space = spaces.Discrete(n=ACTION_SPACE_SIZE, start=0)

        # set env to training or testing mode
        requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/update-env?action={'train' if training else 'test'}&")

    def reset(self):
        # reset the environment to its initial state
        response = requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/reset")
        data = response.json()
        assert "state" in data, "'state' key not in the JSON returned by the server D:"
        state = tensor(data["state"], dtype=float32).view(STATE_SPACE_SIZE * 3, -1)
        return state

    def step(self, action):
        # take a step in the environment based on the given action
        # return the next observation, reward, done flag, and any additional information
        response = requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/step?action={action}&")
        data = response.json()
        for key in ["reward", "isTerminal", "state"]:
            assert key in data, f"'{key}' key not in the JSON returned by the server D:"

        state = tensor(data["state"], dtype=float32).view(STATE_SPACE_SIZE * 3, -1)
        reward = data["reward"]
        done = truncated = data["isTerminal"]
        info = {}
        return state, reward, done, truncated, info

    def render(self, mode='human'):
        # render the current state of the meshenv (optional)
        pass

    def close(self):
        # clean up resources (optional)
        # save mesh state, shutdown c++ server
        requests.get(f"http://{MESH_SERVER_HOST}:{MESH_SERVER_PORT}/bye")
