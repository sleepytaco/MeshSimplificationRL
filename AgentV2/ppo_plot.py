import numpy as np
import torch
from stable_baselines3 import PPO
from AgentV2.custom_env_v2 import MeshEnv
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm

num_tests = 1  # num tests to avg the results over
final_face_count = 200
num_test_steps = (500 - final_face_count) // 2

# pass to PPO model or model.load
kwargs2layer = {
    "path": "ppo_mesh_simplify_10M_2layer", # path to saved model
    "n_steps": 1024,
    "policy": "MlpPolicy",
    "policy_kwargs":dict(activation_fn=nn.ReLU, net_arch=dict(pi=[128, 128], vf=[128, 128])),
    "verbose": 2,
    "gamma": 1,  # default 0.99
    "batch_size": 512,  # must be a factor of max_steps for PPO
    "tensorboard_log": "./ppo_logs_2layer/"
}
kwargs3layer = {
    "path": "ppo_mesh_simplify_1M_3layer", # path to saved model
    "n_steps": 1024,
    "policy": "MlpPolicy",
    "policy_kwargs":dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
    "verbose": 2,
    "gamma": 1,  # default 0.99
    "batch_size": 512,  # must be a factor of max_steps for PPO
    "tensorboard_log": "./ppo_logs_3layer/"
}

mesh_file = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/centaur/test/T156.obj"
env = MeshEnv(mesh_files=[mesh_file], final_face_count=final_face_count, training=False, version=2)
model = PPO.load(env=env, **kwargs2layer)

agent_QEM_costs = np.zeros((num_tests, num_test_steps))
greedy_QEM_costs = np.zeros((num_tests, num_test_steps))
random_agent_QEM_costs = np.zeros((num_tests, num_test_steps))

info = {}
for i in tqdm.tqdm(range(num_tests)):
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)

        obs, rewards, done, trunc, info = env.step(action)
        if done:
            break
    info = env.close()  # impt to call this to save the mesh and return stats for episode
    num_steps = len(info["agentQEMCostsList"])
    agent_QEM_costs[i, :num_steps] = np.array(info["agentQEMCostsList"])
    greedy_QEM_costs[i, :num_steps] = np.array(info["greedyQEMCostsList"])
    random_agent_QEM_costs[i, :num_steps] = np.array(info["randomQEMCostsList"])

# take avg over all the test runs
agent_QEM_costs = np.mean(agent_QEM_costs, axis=0)
greedy_QEM_costs = np.mean(greedy_QEM_costs, axis=0)
random_agent_QEM_costs = np.mean(random_agent_QEM_costs, axis=0)

######## agent simplify 500faces -> 200faces averaged over 100 runs
agentQEMCostsList = agent_QEM_costs
greedyQEMCostsList = greedy_QEM_costs
randomAgentQEMCostsList = random_agent_QEM_costs

plt.plot(agentQEMCostsList, label='RL Agent QEM Costs')
plt.plot(greedyQEMCostsList, label='Greedy Agent QEM Costs')
plt.plot(randomAgentQEMCostsList, label='Random Agent QEM Costs')

plt.xlabel('Valid Edge Collapse Steps')
plt.ylabel(f'Avg Energy Change w.r.t Original Mesh ({num_tests} runs)')
plt.title('RL Agent vs Greedy Agent vs Random Agent Energy Metric')
plt.legend()

plt.show()
###############################