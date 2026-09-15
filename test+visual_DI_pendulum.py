import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def main():
    print("\n--- Caricamento modello e visualizzazione del controllo ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "ppo_double_pendulum_parallel")

    test_env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
    trained_model = PPO.load(model_path)

    obs, info = test_env.reset()


    viewer = test_env.unwrapped.mujoco_renderer.viewer
    if viewer is not None:
        viewer.cam.distance = 6.5      # Allontana la visuale (default è ~3.5)
        viewer.cam.elevation = -10.0   # Leggera vista dall'alto verso il basso
        viewer.cam.azimuth = 90.0      # 90° = vista frontale perfetta
        viewer.cam.lookat[0] = 0.0     # Centro binario X
        viewer.cam.lookat[1] = 0.0     # Centro Y
        viewer.cam.lookat[2] = 0.8     # Alza il centro Z all'altezza delle aste

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