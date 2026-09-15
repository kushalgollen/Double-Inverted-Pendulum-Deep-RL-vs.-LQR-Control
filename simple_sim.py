import gymnasium as gym

# create the environment
env = gym.make("InvertedDoublePendulum-v5", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(1000_000_000):
    # example of a random action
    action = env.action_space.sample()
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()