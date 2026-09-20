import gymnasium as gym
from stable_baselines3 import PPO
import time
import os

def main():
    print("\n--- Initialising training with PPO ---")
    
    # fast with no graphics rendering, for training
    train_env = gym.make("InvertedDoublePendulum-v5")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models/ppo_double_pendulum")

    # defining the PPO model with MLP policy
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1
    )

    TOTAL_TIMESTEPS = 150_000    # from 150k to 400k timesteps, the model can learn to stabilize the pendulum in most cases
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save(model_path)
    train_env.close()

    print(f"--- Model saved succesfully in '{model_path}.zip' ---")

if __name__ == "__main__":
    main()