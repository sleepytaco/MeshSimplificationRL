import numpy as np
from sb3_contrib import TRPO
from AgentV1.custom_env import MeshEnv
import matplotlib.pyplot as plt
import tqdm

num_tests = 10  # num tests to avg the results over
final_face_count = 150
num_test_steps = (500 - final_face_count) // 2

mesh_file = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/centaur/test/T156.obj"
env = MeshEnv(mesh_files=[mesh_file], final_face_count=final_face_count, training=False)
model = TRPO.load(f"trpo_mesh_simplify_11M")

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

plt.plot(agentQEMCostsList, label='RL Agent')
plt.plot(greedyQEMCostsList, label='Greedy Agent')
plt.plot(randomAgentQEMCostsList, label='Random Agent')

plt.xlabel('Valid Edge Collapse Steps')
plt.ylabel(f'Avg Energy Change w.r.t Original Mesh')
plt.title('RL Agent vs Greedy Agent vs Random Agent Energy Metric')
plt.legend()

plt.show()
###############################