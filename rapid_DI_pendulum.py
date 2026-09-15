import gymnasium as gym

# Creazione dell'ambiente con rendering visivo
env = gym.make("InvertedDoublePendulum-v5", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(1000_000_000):
    # Esempio con azione casuale continua (forza applicata al carrello)
    action = env.action_space.sample()
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()