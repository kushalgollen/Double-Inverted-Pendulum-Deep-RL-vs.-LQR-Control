import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def main():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(script_dir, "models", "ppo_double_pendulum_parallel")

    print("--- Initialing Parallel training with PPO ---")

    # Number of parallel environments depends on the number of CPU cores available. Adjust as needed.
    NUM_ENVS = 4 
    
    # makes a vectorized environment with multiple parallel instances of the InvertedDoublePendulum-v5 environment
    train_env = make_vec_env(
        "InvertedDoublePendulum-v5", 
        n_envs=NUM_ENVS, 
        vec_env_cls=SubprocVecEnv
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1
    )


    TOTAL_TIMESTEPS = 1_000_000     # 1M timesteps for training, can be increased for better performance
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Salvataggio
    model.save(model_path)
    train_env.close()
    print(f"\n--- Model saved successfully in '{model_path}.zip' ---")



if __name__ == "__main__":
    main()