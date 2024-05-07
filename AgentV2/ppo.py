# PPO implementation from SB3 library. I set up a custom gym env class called MeshEnv that is supported by SB3.
# MeshEnv relies on the C++ RL environment I created called "meshenv" which is a API server that the env interfaces with
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from custom_env_v2 import MeshEnv
import time
from stable_baselines3.common.env_checker import check_env

# saved model zip to use
model_zip = "ppo_mesh_simplify_1M_3layer"

isTraining = True
final_face_count = 200
usedSavedModelForTraining = True

# pass to model.learn
log_name = "ppo_qem_run_3layer_policy"  # "ppo_approx_error_run",
timesteps = 1_000_000
checkpoint_callback = CheckpointCallback(
    save_freq=500_000,  # Save a checkpoint every 1000 steps
    save_path="./ppo_logs_3layer/",
    name_prefix="ppo_rl_model",
)

# pass to PPO model or model.load
kwargs = {
    "n_steps": 512,
    "policy": "MlpPolicy",
    "policy_kwargs": dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
    "verbose": 2,
    "gamma": 0.999,  # default 0.99
    "batch_size": 512,  # must be a factor of max_steps for PPO
    "tensorboard_log": "./ppo_logs_3layer/"
}

if isTraining:
    print("Training...")
    env = MeshEnv(mesh_files=[
        "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/rabbit/test/T73.obj"],
                  final_face_count=final_face_count, training=isTraining, version=2)

    # sanity check if my custom env follows the Gym interface that SB3 supports and output additional warnings if needed
    check_env(env)

    start_time = time.time()

    if usedSavedModelForTraining:
        print("Using saved model...", model_zip)
        model = PPO.load(model_zip, env=env, **kwargs)
    else:
        model = PPO(env=env, **kwargs)

    model.learn(total_timesteps=timesteps, progress_bar=True, tb_log_name=log_name, callback=checkpoint_callback)
    model.save(f"ppo_mesh_simplify_2M_3layer")

    env.close()

    # policy = model.policy
    # mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Time taken to train for {timesteps} timesteps:", round(execution_time / 60, 2), "mins")
    # time taken for 5M env steps --- 25737s / 428.95 mins / 7 hours 8 minutes 57 seconds
else:
    print("Testing...")
    start_time = time.time()

    mesh_file = "/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/centaur/test/T156.obj"
    env = MeshEnv(mesh_files=[mesh_file], final_face_count=final_face_count, training=isTraining, version=2)
    print("Created env with test file", mesh_file)

    model = PPO.load(model_zip, env=env, **kwargs)
    print("Loaded saved model", model_zip)

    obs, info = env.reset()
    print(info)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, trunc, info = env.step(action)
        if done:
            break
    info = env.close()  # impt to call this to save the mesh
    env.plot_qem_costs(info)
