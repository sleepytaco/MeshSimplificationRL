import numpy as np
import torch
from stable_baselines3 import PPO
from AgentV1.custom_env import MeshEnv
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm

num_tests = 100  # num tests to avg the results over
final_face_count = 200
num_test_steps = (500 - final_face_count) // 2
policy_kwargs = dict(activation_fn=nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

mesh_file = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/centaur/test/T156.obj"
env = MeshEnv(mesh_files=[mesh_file], final_face_count=final_face_count, training=False)
model = PPO.load(f"ppo_mesh_simplify_14M", policy="MlpPolicy", policy_kwargs=policy_kwargs)

agent_QEM_costs = np.zeros((num_tests, num_test_steps))
greedy_QEM_costs = np.zeros((num_tests, num_test_steps))
random_agent_QEM_costs = np.zeros((num_tests, num_test_steps))

info = {}
for i in tqdm.tqdm(range(num_tests)):
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)

        obs = torch.tensor(obs).reshape((1, -1))

        dis = model.policy.get_distribution(obs)
        probs = dis.distribution.probs.detach().numpy().reshape((-1))

        valid_edge_ids = info["validEdgeIds"]
        # print(valid_edge_ids)
        clipped_arr = probs[valid_edge_ids]
        probs = clipped_arr / np.sum(clipped_arr)
        # print(len(probs))

        sample_action = np.random.choice(len(probs), p=probs)
        action = valid_edge_ids[sample_action]

        obs, rewards, done, trunc, info = env.step(action)
        if done:
            break
    info = env.close()  # impt to call this to save the mesh and return stats for episode
    agent_QEM_costs[i] = np.array(info["agentQEMCostsList"])
    greedy_QEM_costs[i] = np.array(info["greedyQEMCostsList"])
    random_agent_QEM_costs[i] = np.array(info["randomQEMCostsList"])

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