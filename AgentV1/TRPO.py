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
usedSavedModelForTraining = True

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
    if usedSavedModelForTraining:
        model = TRPO.load("trpo_mesh_simplify_10M", env=env, verbose=2)
    else:
        model = TRPO("MlpPolicy", env, verbose=2)

    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save("trpo_mesh_simplify_11M")

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
    model = TRPO.load("trpo_mesh_simplify_11M")
    start_time = time.time()

    mesh_path = "meshes/T156.obj"  # "meshes/camel/test/T536.obj"
    env = MeshEnv(mesh_files=[mesh_path], final_face_count=final_face_count,
                          training=isTraining)
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, trunc, info = env.step(action)
        # vec_env.render("human")
        if dones:
            break

    info = env.close()  # impt to call this to save the mesh
    env.plot_qem_costs(info)
