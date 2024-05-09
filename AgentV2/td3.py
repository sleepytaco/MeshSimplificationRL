# DDPG implementation from SB3 library. I set up a custom gym env class called MeshEnv that is supported by SB3.
# MeshEnv relies on the C++ RL environment I created called "meshenv" which is a API server that the env interfaces with
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from custom_env_v2 import MeshEnv
import time
from stable_baselines3.common.env_checker import check_env

# saved model zip to use
model_zip = "sac_mesh_simplify_10M_2layer"

isTraining = True
final_face_count = 200
usedSavedModelForTraining = False

# pass to model.learn
log_name = "sac_qem_run_2layer_policy"  # "sac_approx_error_run",
timesteps = 2_000_000
checkpoint_callback = CheckpointCallback(
    save_freq=1_000_000,  # Save a checkpoint every 1000 steps
    save_path="./sac_logs_2layer/",
    name_prefix="sac_rl_model",
)

# pass to DDPG model or model.load
kwargs2layer = {
    "policy": "MlpPolicy",
    # "policy_kwargs": dict(activation_fn=nn.ReLU, net_arch=dict(pi=[128, 128],qf=[128, 128])), using default one of [400, 300] units works wondders
    "verbose": 2,
    "gamma": 1, # 0.999,  # default 0.99
    "batch_size": 128,  # --- 256 had worked well (but was slow 26 it/s) --- will try 128
    "tensorboard_log": "./sac_logs_2layer/",
    "buffer_size": 96_000,
}
kwargs = kwargs2layer

if isTraining:
    print("Training...")
    env = MeshEnv(mesh_files="/Users/mohammedk/Documents/Brown/CS2951F/MeshSimplificationRL/AgentV1/meshes/train",
                  final_face_count=final_face_count, training=isTraining, version=2)

    # sanity check if my custom env follows the Gym interface that SB3 supports and output additional warnings if needed
    check_env(env)

    start_time = time.time()

    if usedSavedModelForTraining:
        print("Using saved model...", model_zip)
        model = SAC.load(model_zip, env=env, **kwargs)
    else:
        model = SAC(env=env, **kwargs)

    model.learn(total_timesteps=timesteps, progress_bar=True, tb_log_name=log_name, callback=checkpoint_callback)
    model.save(f"sac_mesh_simplify_10M_2layer")

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

    model = SAC.load(model_zip, env=env, **kwargs)
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
