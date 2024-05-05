# TRPO implementation from SB3 library. I set up a custom gym env class called MeshEnv that is supported by SB3.
# MeshEnv relies on the C++ RL environment I created called "meshenv" which is a API server that the env interfaces with

from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from AgentV1.custom_env import MeshEnv
import time
from stable_baselines3.common.env_checker import check_env
import torch.nn as nn

final_face_count = 150
isTraining = True
usedSavedModelForTraining = False

policy_kwargs = dict(activation_fn=nn.ReLU,
                     net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))

# TODO: use custom feature extractor class in TRPO -- use MESHCNN (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor)
# TODO: use callbacks to log training stats https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
# TODO: Parallel environments ?  might be hard because i only have one server instance,
#  could change it up tho to have multiple servers serving at different ports. Could write a Qt script to have them up in a terminal in bg
if isTraining:
    print("Training...")
    env = MeshEnv(mesh_files="/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/centaur/test",
                  final_face_count=final_face_count,
                  training=isTraining)

    # sanity check if my custom env follows the Gym interface that SB3 supports and output additional warnings if needed
    check_env(env)

    start_time = time.time()

    timesteps = int(1e6)  # 200_000
    model = TRPO.load("trpo_mesh_simplify_10M", env=env) if usedSavedModelForTraining \
        else TRPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save("trpo_mesh_simplify")

    # policy = model.policy
    # mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Time taken to train for {timesteps} timesteps:", round(execution_time / 60, 2), "mins")
    # stats time taken:
    # n = 200,000 env steps --- 10.5 mins
    # n = 10,000,000 env steps --- 12.5 hrs lol
else:
    print("Testing...")
    model = TRPO.load("trpo_mesh_simplify_10M")
    start_time = time.time()

    mesh_path = "meshes/T156.obj"  # "meshes/camel/test/T536.obj"
    env = Monitor(MeshEnv(mesh_files=[mesh_path], final_face_count=final_face_count,
                          training=isTraining))
    obs, info = env.reset()
    print(info)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, trunc, info = env.step(action)
        # vec_env.render("human")
        if dones:
            break

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    # print("mean_reward, std_reward:", mean_reward, std_reward)
    print(info)
    env.close()  # impt to call this to save the mesh

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken to test agent:", execution_time, "seconds")

    # saved_policy = model.policy
    # mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=1) # , deterministic=True)

    import matplotlib.pyplot as plt

    # Sample data
    list1 = info["agentQEMCostsList"]
    list2 = info["greedyQEMCostsList"]
    list3 = info["randomQEMCostsList"]

    # Plotting
    plt.plot(list1, label='RL Agent QEM Costs')
    plt.plot(list2, label='Greedy QEM Costs')
    plt.plot(list3, label='Random QEM Costs')

    # Adding labels and legend
    plt.xlabel('Steps')
    plt.ylabel('QEM Costs')
    plt.title('Plot of RL Agent QEM Costs vs Greedy QEM Costs vs Random Agent QEM Costs')
    plt.legend()

    # Display the plot
    plt.show()