import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "ppo_double_pendulum_parallel")

    # ========================================================
    # 1. TRAINING PARALLELO (sfrutta i core della CPU)
    # ========================================================
    print("--- Inizio Addestramento Parallelo con PPO ---")

    # Numero di ambienti paralleli (8 o 4 in base ai thread della CPU)
    NUM_ENVS = 4 #cambiatooooo
    
    # Crea NUM_ENVS ambienti isolati eseguiti in parallelo su processi separati
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

    # 400.000 passi complessivi (distribuiti sugli 8 ambienti)
    TOTAL_TIMESTEPS = 1_000_000
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Salvataggio
    model.save(model_path)
    train_env.close()
    print(f"\n--- Modello salvato con successo in '{model_path}.zip' ---")

    # ========================================================
    # 2. TEST VISIVO (Singolo ambiente con rendering a schermo)
    # ========================================================
    print("\n--- Caricamento modello e visualizzazione del controllo ---")

    test_env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
    trained_model = PPO.load(model_path)

    obs, info = test_env.reset()

    try:
        while True:
            action, _states = trained_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            # Mantiene il rendering fluido a ~50-60 FPS
            time.sleep(0.01)

            if terminated or truncated:
                obs, info = test_env.reset()

    except KeyboardInterrupt:
        print("\nSimulazione interrotta dall'utente.")
    finally:
        test_env.close()

if __name__ == "__main__":
    main()