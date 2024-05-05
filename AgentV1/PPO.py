# PPO implementation from SB3 library. I set up a custom gym env class called MeshEnv that is supported by SB3.
# MeshEnv relies on the C++ RL environment I created called "meshenv" which is a API server that the env interfaces with
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from AgentV1.custom_env import MeshEnv
from gymnasium import spaces
import time
from stable_baselines3.common.env_checker import check_env
from AgentV1.settings import *
from stable_baselines3.common.evaluation import evaluate_policy

import torch
import torch.nn as nn
import torch.nn.functional as F

final_face_count = 150
isTraining = True
usedSavedModelForTraining = True

timesteps = 1_000_000#_000
policy_kwargs = dict(activation_fn=nn.ReLU,
                     net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))

# TODO: use custom feature extractor class in TRPO -- use MESHCNN (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor)
# TODO: use callbacks to log training stats https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
if isTraining:
    print("Training...")
    env = MeshEnv(mesh_files=["/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/centaur/test/T156.obj"],
                  final_face_count=final_face_count, training=isTraining)

    # sanity check if my custom env follows the Gym interface that SB3 supports and output additional warnings if needed
    check_env(env)

    start_time = time.time()

    if usedSavedModelForTraining and os.path.exists(f"ppo_mesh_simplify_{timesteps}"):
        model = PPO.load(f"ppo_mesh_simplify_{timesteps}", env=env, policy_kwargs=policy_kwargs, verbose=2, n_steps=1024, batch_size=512)
    else:
        model = PPO(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=2, n_steps=1024, batch_size=512)

    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(f"ppo_mesh_simplify_{timesteps}")

    env.close()

    # policy = model.policy
    # mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Time taken to train for {timesteps} timesteps:", round(execution_time / 60, 2), "mins")
    # stats time taken:
    # n = 100, 000 env steps --- 5.39 mins
    # n = 200,000 env steps --- 10.5 mins
    # n = 1, 000, 000 env steps ---
    # n = 10,000,000 env steps --- 12.5 hrs lol
else:
    print("Testing...")
    start_time = time.time()

    mesh_file = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/centaur/test/T156.obj"
    env = MeshEnv(mesh_files=[mesh_file], final_face_count=final_face_count, training=isTraining)
    model = PPO.load(f"ppo_mesh_simplify_1000000", policy_kwargs=policy_kwargs)

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, trunc, info = env.step(action)
        if done:
            break

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    # print("mean_reward, std_reward:", mean_reward, std_reward)

    env.close()  # impt to call this to save the mesh

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken to test agent:", execution_time, "seconds")

    env.plot_qem_costs(info)