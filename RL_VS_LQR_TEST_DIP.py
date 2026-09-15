import gymnasium as gym
import numpy as np
import scipy.linalg
from stable_baselines3 import PPO

# ==============================================================================
# 1. DEFINIZIONE DEL CONTROLLORE LQR
# ==============================================================================
def compute_lqr_gain():
    # Parametri fisici approssimati di InvertedDoublePendulum-v5
    M = 1.0     # Carrello
    m1, m2 = 0.1, 0.1
    l1, l2 = 0.6, 0.6
    d1, d2 = 0.3, 0.3
    g = 9.81
    J1 = (1.0 / 12.0) * m1 * (l1**2)
    J2 = (1.0 / 12.0) * m2 * (l2**2)

    M_mat = np.array([
        [M + m1 + m2,        m1*d1 + m2*l1,           m2*d2],
        [m1*d1 + m2*l1,      J1 + m1*(d1**2) + m2*(l1**2), m2*l1*d2],
        [m2*d2,              m2*l1*d2,                J2 + m2*(d2**2)]
    ])
    K_mat = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -(m1*d1 + m2*l1)*g, 0.0],
        [0.0, 0.0, -m2*d2*g]
    ])
    H_mat = np.array([[1.0], [0.0], [0.0]])
    M_inv = np.linalg.inv(M_mat)

    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = -np.dot(M_inv, K_mat)

    B = np.zeros((6, 1))
    B[3:6, :] = np.dot(M_inv, H_mat)

    # Matrici di costo
    Q = np.diag([20.0, 100.0, 150.0, 5.0, 20.0, 30.0])
    R = np.array([[0.01]])

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ P)
    return K.flatten()

def get_lqr_action(K, env):
    # Estrae lo stato fisico esatto da MuJoCo: [x, th1, th2, dx, dth1, dth2]
    qpos = env.unwrapped.data.qpos
    qvel = env.unwrapped.data.qvel
    
    state = np.array([qpos[0], qpos[1], qpos[2], qvel[0], qvel[1], qvel[2]])
    u = np.dot(K, state)
    
    # Clip dell'azione entro i limiti dell'action space dell'ambiente
    return np.clip(np.array([u], dtype=np.float32), -10.0, 10.0)


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