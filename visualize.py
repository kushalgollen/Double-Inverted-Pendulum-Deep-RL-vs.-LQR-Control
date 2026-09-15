import os
import time
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    print("\n--- Laoding and visualising model ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models/ppo_double_pendulum_parallel")

    test_env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
    trained_model = PPO.load(model_path)

    obs, info = test_env.reset()


    viewer = test_env.unwrapped.mujoco_renderer.viewer
    if viewer is not None:
        viewer.cam.distance = 6.5      
        viewer.cam.elevation = -10.0   
        viewer.cam.azimuth = 90.0      
        viewer.cam.lookat[0] = 0.0     
        viewer.cam.lookat[1] = 0.0     
        viewer.cam.lookat[2] = 0.8     

    try:
        while True:

            action, _states = trained_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            # have a small delay to make the simulation visually smoother
            time.sleep(0.01)

            if terminated or truncated:
                obs, info = test_env.reset()

    except KeyboardInterrupt:
        print("\nSimulation interupted by user.")
    finally:
        test_env.close()

if __name__ == "__main__": 
    main()