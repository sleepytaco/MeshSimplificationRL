# PPO implementation from SB3 library. I set up a custom gym env class called MeshEnv that is supported by SB3.
# MeshEnv relies on the C++ RL environment I created called "meshenv" which is a API server that the env interfaces with
import os

import numpy as np
from stable_baselines3 import PPO
from AgentV1.custom_env import MeshEnv
import time
from stable_baselines3.common.env_checker import check_env
import torch.nn as nn

final_face_count = 150
isTraining = False
usedSavedModelForTraining = True

timesteps = 1_000_000
policy_kwargs = dict(activation_fn=nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
gamma = 0.99
verbose = 2

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
        print("Using saved model...")
        model = PPO.load(f"ppo_mesh_simplify_10M", env=env, policy_kwargs=policy_kwargs, verbose=verbose)
    else:
        model = PPO(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(f"ppo_mesh_simplify_11M")

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
    model = PPO.load(f"ppo_mesh_simplify_10M", policy="MlpPolicy", policy_kwargs=policy_kwargs, gamma=gamma)

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, trunc, info = env.step(action)
        if done:
            break
    info = env.close()  # impt to call this to save the mesh
    env.plot_qem_costs(info)
