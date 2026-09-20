import gymnasium as gym
import numpy as np
import scipy.linalg
from stable_baselines3 import PPO
import mujoco
import os

def compute_lqr_gain():
    temp_env = gym.make("InvertedDoublePendulum-v5")
    model = temp_env.unwrapped.model
    data = temp_env.unwrapped.data
    
    
    frame_skip = temp_env.unwrapped.frame_skip
    sim_dt = model.opt.timestep
    dt_env = sim_dt * frame_skip  # real time decision making (0.05s o 0.02s)

    orig_integrator = model.opt.integrator
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    nx = 6
    nu = 1

    A_sim = np.zeros((nx, nx), dtype=np.float64, order="C")
    B_sim = np.zeros((nx, nu), dtype=np.float64, order="C")
    
    mujoco.mjd_transitionFD(model, data, 1e-6, 1, A_sim, B_sim, None, None)
    model.opt.integrator = orig_integrator
    temp_env.close()

    # conversion timing:
    
    A_c = (A_sim - np.eye(nx)) / sim_dt
    B_c = B_sim / sim_dt

    A_env = np.eye(nx) + A_c * dt_env
    B_env = B_c * dt_env

    # LQR param 
    Q = np.diag([
        5.0,     # x (cart)
        40.0,    # th1 (angle 1)
        60.0,    # th2 (angle 2)
        5.0,     # x_dot (vel cart)
        40.0,    # th1_dot (l1 damp - CRUCIAL)
        60.0     # th2_dot (l2 damp - CRUCIAL)
    ])
    R = np.array([[2.0]])

    P = scipy.linalg.solve_discrete_are(A_env, B_env, Q, R)
    K = np.linalg.inv(R + B_env.T @ P @ B_env) @ (B_env.T @ P @ A_env)

    return K.flatten()

def get_lqr_action(K, env):
    # 1. extract state from the environment
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    
    # State: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
    state = np.concatenate([qpos, qvel])
    
    # 2. Lqr control law: u = -K * state
    u = -float(np.dot(K, state))
    
    # 3. Clip the action to the valid range of the environment
    action = np.clip(np.array([u], dtype=np.float32), -1.0, 1.0)
    return action



def evaluate_controller(controller_type, model_ppo, K_lqr, num_episodes=50, impulse_test=False):
    env = gym.make("InvertedDoublePendulum-v5")
    
    total_ise = []
    total_energy = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep) # same seed for reproducibility
        ep_ise = 0.0
        ep_energy = 0.0
        steps = 0

        terminated, truncated = False, False
        while not (terminated or truncated):
            # Test for impulse disturbance at step 200
            if impulse_test and steps == 200:
                env.unwrapped.data.qvel[0] += 0.8  # giving a push to the cart (0.8 m/s is a mild push)

            if controller_type == "RL":
                action, _ = model_ppo.predict(obs, deterministic=True)
            else:
                action = get_lqr_action(K_lqr, env)

            obs, reward, terminated, truncated, _ = env.step(action)

       
            qpos = env.unwrapped.data.qpos
           
            x, th1, th2 = qpos[0], qpos[1], qpos[2]

    

            ep_ise += (th1**2 + th2**2 + 0.1 * (x**2))
            ep_energy += float(action[0]**2)
            steps += 1

        total_ise.append(ep_ise / max(1, steps))
        total_energy.append(ep_energy / max(1, steps))
        episode_lengths.append(steps)

    env.close()
    return {
        "mean_len": np.mean(episode_lengths),
        "mean_ise": np.mean(total_ise),
        "mean_energy": np.mean(total_energy),
        "survival_rate": np.mean([1.0 if l == 1000 else 0.0 for l in episode_lengths]) * 100.0
    }


# running the benchmark when the script is executed directly

if __name__ == "__main__":
    print("--- Init. Benchmark RL vs LQR ---")
    
    # 1. loading the trained PPO model
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(script_dir, "models", "ppo_double_pendulum_parallel")
    model_ppo = PPO.load(model_path)
    
    # 2. Calculate LQR gains
    K_lqr = compute_lqr_gain()
    print(f"LQR Gains: {np.round(K_lqr, 2)}\n")

    # TEST 1: Nominal Conditions (50 Episodes)
    print("=== TEST 1: Nominal Conditions (50 Episodes) ===")
    res_lqr_nom = evaluate_controller("LQR", model_ppo, K_lqr, num_episodes=50, impulse_test=False)
    res_rl_nom = evaluate_controller("RL", model_ppo, K_lqr, num_episodes=50, impulse_test=False)

    print(f"{'Metrics':<25} | {'LQR':<15} | {'RL (PPO)':<15}")
    print("-" * 60)
    print(f"{'Survival Rate (%)':<25} | {res_lqr_nom['survival_rate']:<15.1f} | {res_rl_nom['survival_rate']:<15.1f}")
    print(f"{'Mean ISE (Error)':<25} | {res_lqr_nom['mean_ise']:<15.4f} | {res_rl_nom['mean_ise']:<15.4f}")
    print(f"{'Mean Energy (u^2)':<25} | {res_lqr_nom['mean_energy']:<15.4f} | {res_rl_nom['mean_energy']:<15.4f}")

    # TEST 2: Impulse Disturbance Stress-Test (Push at step 200)
    print("\n=== TEST 2: Impulse Disturbance Stress-Test (Push at step 200) ===")
    res_lqr_dist = evaluate_controller("LQR", model_ppo, K_lqr, num_episodes=50, impulse_test=True)
    res_rl_dist = evaluate_controller("RL", model_ppo, K_lqr, num_episodes=50, impulse_test=True)

    print(f"{'Metrics':<25} | {'LQR':<15} | {'RL (PPO)':<15}")
    print("-" * 60)
    print(f"{'Survival Rate (%)':<25} | {res_lqr_dist['survival_rate']:<15.1f} | {res_rl_dist['survival_rate']:<15.1f}")
    print(f"{'Mean ISE (Error)':<25} | {res_lqr_dist['mean_ise']:<15.4f} | {res_rl_dist['mean_ise']:<15.4f}")
    print(f"{'Mean Energy (u^2)':<25} | {res_lqr_dist['mean_energy']:<15.4f} | {res_rl_dist['mean_energy']:<15.4f}")





















    