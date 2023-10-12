import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from ray.rllib.algorithms.algorithm import Algorithm
import os

env_name = "Pusher-v4"
env = gym.make(env_name, render_mode="rgb_array_list")
env = gym.wrappers.RecordEpisodeStatistics(env)
algo = Algorithm.from_checkpoint("/root/ray_results/pusher-ppo/PPO_Pusher-v4_54a06_00000_0_2023-09-26_05-19-56/checkpoint_003560")

base_video_folder = "/workspaces/Pusher-v4/Pusher/video_output"

for episode in range(10):
    episode_video_folder = os.path.join(base_video_folder, f'episode_{episode + 1}')
    os.makedirs(episode_video_folder, exist_ok=True)
    
    env = RecordVideo(env, episode_video_folder)

    episode_reward = 0
    terminated = truncated = False

    obs, info = env.reset()

    while not terminated and not truncated:
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

env.close()
