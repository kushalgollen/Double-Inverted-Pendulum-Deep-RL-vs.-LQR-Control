import gymnasium as gym
import numpy as np
import scipy.linalg
from stable_baselines3 import PPO
import mujoco

# ==============================================================================
# 1. DEFINIZIONE DEL CONTROLLORE LQR
# ==============================================================================

def compute_lqr_gain():
    temp_env = gym.make("InvertedDoublePendulum-v5")
    model = temp_env.unwrapped.model
    data = temp_env.unwrapped.data
    
    # Rilevamento automatico del timestep effettivo dell'ambiente (dt * frame_skip)
    frame_skip = temp_env.unwrapped.frame_skip
    sim_dt = model.opt.timestep
    dt_env = sim_dt * frame_skip  # Tempo reale tra una decisione e la successiva (0.05s o 0.02s)

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

    # Conversione a tempo continuo esatto:
    # A_c = (A_sim - I) / sim_dt, B_c = B_sim / sim_dt
    A_c = (A_sim - np.eye(nx)) / sim_dt
    B_c = B_sim / sim_dt

    # Discretizzazione su dt_env reale (zero-order hold approssimato al 1° ordine)
    A_env = np.eye(nx) + A_c * dt_env
    B_env = B_c * dt_env

    # Pesi LQR con ALTO smorzamento sulle velocità per fermare le oscillazioni
    # Stato: [x, th1, th2, x_dot, th1_dot, th2_dot]
    Q = np.diag([
        5.0,     # x (carrello)
        40.0,    # th1 (angolo 1)
        60.0,    # th2 (angolo 2)
        5.0,     # x_dot (velocità carrello)
        40.0,    # th1_dot (smorzamento asta 1 - CRUCIALE)
        60.0     # th2_dot (smorzamento asta 2 - CRUCIALE)
    ])
    R = np.array([[2.0]])

    P = scipy.linalg.solve_discrete_are(A_env, B_env, Q, R)
    K = np.linalg.inv(R + B_env.T @ P @ B_env) @ (B_env.T @ P @ A_env)

    return K.flatten()

def get_lqr_action(K, env):
    # 1. Estrazione diretta dello stato esatto da MuJoCo
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    
    # Stato: [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
    state = np.concatenate([qpos, qvel])
    
    # 2. Legge LQR canonica: u = -K * x
    u = -float(np.dot(K, state))
    
    # 3. Clip dell'azione nello spazio ammesso dall'ambiente Gymnasium ([-1.0, 1.0])
    action = np.clip(np.array([u], dtype=np.float32), -1.0, 1.0)
    return action


# ==============================================================================
# 2. RUNNER DI VALUTAZIONE
# ==============================================================================
def evaluate_controller(controller_type, model_ppo, K_lqr, num_episodes=50, impulse_test=False):
    env = gym.make("InvertedDoublePendulum-v5")
    
    total_ise = []
    total_energy = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep) # Identico seed per parità di condizioni
        ep_ise = 0.0
        ep_energy = 0.0
        steps = 0

        terminated, truncated = False, False
        while not (terminated or truncated):
            # Test di disturbo impulsivo allo step 200
            if impulse_test and steps == 200:
                env.unwrapped.data.qvel[0] += 0.8  # Forte calcio al carrello (+3.5 m/s)

            if controller_type == "RL":
                action, _ = model_ppo.predict(obs, deterministic=True)
            else:
                action = get_lqr_action(K_lqr, env)

            obs, reward, terminated, truncated, _ = env.step(action)

            # Telemetria dello stato
            qpos = env.unwrapped.data.qpos
            #qvel = env.unwrapped.data.qvel
            x, th1, th2 = qpos[0], qpos[1], qpos[2]

            #if ep == 0 and controller_type == "LQR":
             #   print(f"Step {steps:03d} | x: {qpos[0]:+.2f} | th1: {qpos[1]:+.2f} | th2: {qpos[2]:+.2f} | act: {action[0]:+.2f}")

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


# ==============================================================================
# 3. ESECUZIONE DEL BENCHMARK
# ==============================================================================
if __name__ == "__main__":
    print("--- Avvio Benchmark RL vs LQR ---")
    
    # 1. Carica modello RL
    model_ppo = PPO.load("ppo_double_pendulum_parallel")
    
    # 2. Calcola guadagni LQR
    K_lqr = compute_lqr_gain()
    print(f"Guadagni LQR calcolati: {np.round(K_lqr, 2)}\n")

    # TEST 1: Condizioni standard di equilibrio (50 episodi)
    print("=== TEST 1: Condizioni Nominali (50 Episodi) ===")
    res_lqr_nom = evaluate_controller("LQR", model_ppo, K_lqr, num_episodes=50, impulse_test=False)
    res_rl_nom = evaluate_controller("RL", model_ppo, K_lqr, num_episodes=50, impulse_test=False)

    print(f"{'Metrica':<25} | {'LQR':<15} | {'RL (PPO)':<15}")
    print("-" * 60)
    print(f"{'Survival Rate (%)':<25} | {res_lqr_nom['survival_rate']:<15.1f} | {res_rl_nom['survival_rate']:<15.1f}")
    print(f"{'ISE medio (Errore)':<25} | {res_lqr_nom['mean_ise']:<15.4f} | {res_rl_nom['mean_ise']:<15.4f}")
    print(f"{'Energia media (u^2)':<25} | {res_lqr_nom['mean_energy']:<15.4f} | {res_rl_nom['mean_energy']:<15.4f}")

    # TEST 2: Reiezione dei disturbi impulsivi (Spinta di 3.5 m/s)
    print("\n=== TEST 2: Stress-Test Disturbo Impulsivo (Spinta a step 200) ===")
    res_lqr_dist = evaluate_controller("LQR", model_ppo, K_lqr, num_episodes=50, impulse_test=True)
    res_rl_dist = evaluate_controller("RL", model_ppo, K_lqr, num_episodes=50, impulse_test=True)

    print(f"{'Metrica':<25} | {'LQR':<15} | {'RL (PPO)':<15}")
    print("-" * 60)
    print(f"{'Survival Rate (%)':<25} | {res_lqr_dist['survival_rate']:<15.1f} | {res_rl_dist['survival_rate']:<15.1f}")
    print(f"{'ISE medio (Errore)':<25} | {res_lqr_dist['mean_ise']:<15.4f} | {res_rl_dist['mean_ise']:<15.4f}")
    print(f"{'Energia media (u^2)':<25} | {res_lqr_dist['mean_energy']:<15.4f} | {res_rl_dist['mean_energy']:<15.4f}")





















    