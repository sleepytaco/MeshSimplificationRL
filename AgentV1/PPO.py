# PPO implementation from SB3 library. I set up a custom gym env class called MeshEnv that is supported by SB3.
# MeshEnv relies on the C++ RL environment I created called "meshenv" which is a API server that the env interfaces with
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from AgentV1.custom_env import MeshEnv
import time
from stable_baselines3.common.env_checker import check_env
import torch.nn as nn

final_face_count = 150
isTraining = True
usedSavedModelForTraining = True

timesteps = 1_000_000
policy_kwargs = dict(activation_fn=nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
verbose = 2
gamma = 1  # default 0.99

# TODO: use custom feature extractor class in PPO -- use MESHCNN (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor)
# TODO: use callbacks to log training stats https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
if isTraining:
    print("Training...")
    env = MeshEnv(mesh_files="/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/train",
                  final_face_count=final_face_count, training=isTraining)

    # sanity check if my custom env follows the Gym interface that SB3 supports and output additional warnings if needed
    check_env(env)

    start_time = time.time()

    if usedSavedModelForTraining:
        model_zip = "ppo_mesh_simplify_13M"
        print("Using saved model...", model_zip)
        model = PPO.load(model_zip, env=env, policy_kwargs=policy_kwargs, verbose=verbose, gamma=gamma)
    else:
        model = PPO(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(f"ppo_mesh_simplify_14M")

    env.close()

    # policy = model.policy
    # mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Time taken to train for {timesteps} timesteps:", round(execution_time / 60, 2), "mins")
    # stats time taken:
    # n = 100, 000 env steps --- 5.39 mins
    # n = 200,000 env steps --- 10.5 mins
    # n = 1, 000, 000 env steps --- 62 mins
    # n = 10,000,000 env steps --- 12.5 hrs lol
else:
    print("Testing...")
    start_time = time.time()

    mesh_file = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/centaur/test/T156.obj"
    env = MeshEnv(mesh_files=[mesh_file], final_face_count=final_face_count, training=isTraining)
    model = PPO.load(f"ppo_mesh_simplify_14M", policy="MlpPolicy", policy_kwargs=policy_kwargs)

    obs, info = env.reset()
    print(info)
    while True:

        action, _states = model.predict(obs)
        # print("-----")
        # print("sb3 action:", action)
        # obs = torch.tensor(obs).reshape((1, -1))
        #
        # dis = model.policy.get_distribution(obs)
        # probs = dis.distribution.probs.detach().numpy().reshape((-1))
        # print("argmax action", np.argmax(probs))
        # valid_edge_ids = info["validEdgeIds"]
        # # print(valid_edge_ids)
        # clipped_arr = probs[valid_edge_ids]
        # probs = clipped_arr / np.sum(clipped_arr)
        # #print(len(probs))
        #
        # sample_action = np.random.choice(len(probs), p=probs)
        # action = valid_edge_ids[sample_action]
        # print("sampled action", action)

        obs, rewards, done, trunc, info = env.step(action)
        if done:
            break
    info = env.close()  # impt to call this to save the mesh
    env.plot_qem_costs(info)
