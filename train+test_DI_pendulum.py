import gymnasium as gym
from stable_baselines3 import PPO
import time

# ==========================================
# 1. FASE DI TRAINING (veloce, senza grafica)
# ==========================================
print("--- Inizio Addestramento con PPO ---")

# Creazione dell'ambiente senza render_mode per la massima velocità
train_env = gym.make("InvertedDoublePendulum-v5")

# Definizione del modello PPO con policy MLP (Multi-Layer Perceptron)
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    gamma=0.99,
    verbose=1
)

# 150.000 timesteps sono sufficienti per convergere a una policy stabile
TOTAL_TIMESTEPS = 150_000    # da 150_000 a 400_000 per un addestramento più robusto
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Salvataggio del modello su disco
model_path = "ppo_double_pendulum"
model.save(model_path)
train_env.close()

print(f"--- Modello salvato con successo in '{model_path}.zip' ---")


# ==========================================
# 2. FASE DI TEST (con rendering a schermo)
# ==========================================
print("\n--- Caricamento modello e visualizzazione del controllo ---")

# Riapriamo l'ambiente con rendering grafico attivo
test_env = gym.make("InvertedDoublePendulum-v5", render_mode="human")

# Caricamento del modello salvato
trained_model = PPO.load(model_path)

obs, info = test_env.reset()

# Eseguiamo la simulazione per osservare la stabilità
for step in range(2000):
    # deterministic=True fa usare alla rete l'azione ottimale (senza rumore stocastico)
    action, _states = trained_model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = test_env.step(action)
    
    # Piccolo sleep opzionale per sincronizzare il framerate a 50 FPS (0.02s a passo)
    time.sleep(0.01)
    
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()