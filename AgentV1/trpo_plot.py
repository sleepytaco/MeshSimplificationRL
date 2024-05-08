import numpy as np
from sb3_contrib import TRPO
from custom_env import MeshEnv
import matplotlib.pyplot as plt
import tqdm

num_tests = 1  # num tests to avg the results over
final_face_count = 200
num_test_steps = (500 - final_face_count) // 2

mesh_file = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/centaur/test/T156.obj"
mesh_file = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/rabbit/test/T73.obj"
env = MeshEnv(mesh_files=[mesh_file], final_face_count=final_face_count, training=False)
model = TRPO.load("trpo_mesh_simplify_11M.zip")

agent_QEM_costs = np.zeros((num_tests, num_test_steps))
greedy_QEM_costs = np.zeros((num_tests, num_test_steps))
random_agent_QEM_costs = np.zeros((num_tests, num_test_steps))

agent_energy_approx = np.zeros((num_tests, num_test_steps))
greedy_energy_approx = np.zeros((num_tests, num_test_steps))
random_energy_approx = np.zeros((num_tests, num_test_steps))

agent_non_manifold_collapses = np.zeros((num_tests, num_test_steps))
greedy_non_manifold_collapses = np.zeros((num_tests, num_test_steps))
random_non_manifold_collapses = np.zeros((num_tests, num_test_steps))

info = {}
for i in tqdm.tqdm(range(num_tests)):
    obs, info = env.reset()
    while True:
        action = np.random.choice(range(750))  # pick a random action to simulate random agent
        obs, rewards, done, trunc, info = env.step(action)
        if done:
            break
    info = env.close()  # impt to call this to save the mesh and return stats for episode

    random_agent_QEM_costs[i] = np.array(info["agentQEMCostsList"])
    random_energy_approx[i] = np.array(info["agentEnergyApproxErrors"])
    random_non_manifold_collapses[i] = np.array(info["agentNonManifoldCollapsesList"])

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
    # random_agent_QEM_costs[i] = np.array(info["randomQEMCostsList"])

    agent_energy_approx[i] = np.array(info["agentEnergyApproxErrors"])
    greedy_energy_approx[i] = np.array(info["greedyEnergyApproxErrors"])
    # random_energy_approx[i] = np.array(info["randomEnergyApproxErrors"])

    agent_non_manifold_collapses[i] = np.array(info["agentNonManifoldCollapsesList"])
    greedy_non_manifold_collapses[i] = np.array(info["greedyNonManifoldCollapsesList"])
    # random_non_manifold_collapses[i] = np.array(info["randomNonManifoldCollapsesList"])

#
# print(info)
# print(info["greedyQEMCostsList"])
# print(info["randomQEMCostsList"])
# take avg over all the test runs
agent_QEM_costs = np.mean(agent_QEM_costs, axis=0)
greedy_QEM_costs = np.mean(greedy_QEM_costs, axis=0)
random_agent_QEM_costs = np.mean(random_agent_QEM_costs, axis=0)

agent_energy_approx = np.mean(agent_energy_approx, axis=0)
greedy_energy_approx = np.mean(greedy_energy_approx, axis=0)
random_energy_approx = np.mean(random_energy_approx, axis=0)

agent_non_manifold_collapses = np.mean(agent_non_manifold_collapses, axis=0)
greedy_non_manifold_collapses = np.mean(greedy_non_manifold_collapses, axis=0)
random_non_manifold_collapses = np.mean(random_non_manifold_collapses, axis=0)

######## agent simplify 500faces -> 200faces averaged over 100 runs
# Create three figures
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(5, 15))

# Plotting on the first figure
ax1.plot(agent_QEM_costs, label='RL Agent')
ax1.plot(random_agent_QEM_costs, label='Random Agent')
ax1.plot(greedy_QEM_costs, label='Greedy Agent')
ax1.set_xlabel('Valid Edge Collapse Steps')
ax1.set_ylabel(f'Avg QEM Error per step')
# ax1.set_title('RL vs Greedy vs Random Agent Avg QEM Errors per step')
ax1.legend()

ax2.plot(agent_energy_approx, label='RL Agent')
ax2.plot(random_energy_approx, label='Random Agent')
ax2.plot(greedy_energy_approx, label='Greedy Agent')
ax2.set_xlabel('Valid Edge Collapse Steps')
ax2.set_ylabel(f'Avg Approximation Error w.r.t Original Mesh')
# ax2.set_title('RL vs Greedy vs Random Agent Avg Energy change per step')
ax2.legend()

ax3.plot(agent_non_manifold_collapses, label='RL Agent')
ax3.plot(random_non_manifold_collapses, label='Random Agent')
ax3.plot(greedy_non_manifold_collapses, label='Greedy Agent')
ax3.set_xlabel('Valid Edge Collapse Steps')
ax3.set_ylabel(f'Avg Cumulative Non-manifold Collapses')
# ax3.set_title('RL vs Greedy vs Random Agent Avg Non-manifold collapses per step')
ax3.legend()

# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.subplots_adjust(left=0.15)

# Show the plots
plt.show()

################################
# plt.plot(agent_QEM_costs, label='RL Agent')
# plt.plot(random_agent_QEM_costs, label='Random Agent')
# plt.plot(greedy_QEM_costs, label='Greedy Agent')
#
# plt.plot(agent_energy_approx, label='RL Agent')
# plt.plot(random_energy_approx, label='Random Agent')
# plt.plot(greedy_energy_approx, label='Greedy Agent')
#
# print(agent_non_manifold_collapses)
# print(random_non_manifold_collapses)
# print(greedy_non_manifold_collapses)
#
# plt.plot(agent_non_manifold_collapses, label='RL Agent')
# plt.plot(random_non_manifold_collapses, label='Random Agent')
# plt.plot(greedy_non_manifold_collapses, label='Greedy Agent')
#
# plt.xlabel('Valid Edge Collapse Steps')
# plt.ylabel(f'Avg Non-manifold Edge Collapses')
# plt.title('RL Agent vs Greedy Agent vs Random Agent Energy Metric')
# plt.legend()
#
# plt.show()
###############################