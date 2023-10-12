import gymnasium as gym
import gymnasium
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm

env_name = "Pusher-v4"
env = gym.make(env_name, render_mode="human")

algo = Algorithm.from_checkpoint("/root/ray_results/pusher-ppo/PPO_Pusher-v4_54a06_00000_0_2023-09-26_05-19-56/checkpoint_001610")

episode_reward = 0
terminated = truncated = False

obs, info = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs,explore=False)
    
    print(action)
    print("----------------------------------------------")
    obs, reward, terminated, truncated, info = env.step(action)
    print(info)
    episode_reward += reward