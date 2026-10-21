import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import envs
from envs.landing_env import HoverSlamEnv
from sim.rocket import IDX_THETA, IDX_VY, IDX_Y


def run_random_rollout(seed: int = 42, output_path: str = "results/random_rollout.png"):
    """Run an episode with uniformly sampled random actions and plot trajectory."""
    env = HoverSlamEnv(randomize_start=True)
    obs, info = env.reset(seed=seed)

    times = []
    altitudes = []
    vertical_velocities = []
    throttles = []
    tilts_deg = []
    rewards = []

    dt = env.dt
    total_reward = 0.0
    step_num = 0

    assert env.state is not None
    times.append(0.0)
    altitudes.append(float(env.state[IDX_Y]))
    vertical_velocities.append(float(env.state[IDX_VY]))
    throttles.append(0.0)
    tilts_deg.append(float(np.degrees(env.state[IDX_THETA])))

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        step_num += 1
        total_reward += reward

        assert env.state is not None
        times.append(step_num * dt)
        altitudes.append(float(env.state[IDX_Y]))
        vertical_velocities.append(float(env.state[IDX_VY]))
        throttles.append(env.last_throttle)
        tilts_deg.append(float(np.degrees(env.state[IDX_THETA])))
        rewards.append(reward)

        if terminated or truncated:
            break

    env.close()

    print(f"Random rollout completed in {step_num} steps ({times[-1]:.2f} s).")
    print(f"Outcome: {'Landed successfully' if info['landed'] else 'Crashed / Terminated'}")
    print(f"Final Altitude: {altitudes[-1]:.2f} m | Final Vy: {vertical_velocities[-1]:.2f} m/s")
    print(f"Total Cumulative Reward: {total_reward:.2f}")

    # Plotting trajectory
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig.patch.set_facecolor("#0f172a")

    for ax in axs:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # 1. Altitude
    axs[0].plot(times, altitudes, color="#38bdf8", linewidth=1.8, label="Altitude (m)")
    axs[0].axhline(0.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Ground (y=0)")
    axs[0].set_ylabel("Altitude (m)", color="#f8fafc", fontsize=10)
    axs[0].legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
    axs[0].set_title(f"HoverSlam Random Action Rollout (Total Steps: {step_num})", color="#f8fafc", fontsize=12)

    # 2. Vertical velocity
    axs[1].plot(times, vertical_velocities, color="#f59e0b", linewidth=1.8, label="vy (m/s)")
    axs[1].axhline(0.0, color="#94a3b8", linestyle=":", linewidth=1.0)
    axs[1].axhline(-2.0, color="#22c55e", linestyle="--", linewidth=1.0, label="Soft touchdown limit (-2 m/s)")
    axs[1].set_ylabel("Vertical Vel (m/s)", color="#f8fafc", fontsize=10)
    axs[1].legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # 3. Commanded Throttle
    axs[2].step(times, throttles, color="#10b981", linewidth=1.8, where="post", label="Throttle Fraction")
    axs[2].set_ylabel("Throttle", color="#f8fafc", fontsize=10)
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # 4. Tilt Angle
    axs[3].plot(times, tilts_deg, color="#ec4899", linewidth=1.8, label="Pitch Angle (deg)")
    axs[3].axhline(0.0, color="#94a3b8", linestyle=":", linewidth=1.0)
    axs[3].axhline(5.0, color="#ef4444", linestyle="--", linewidth=1.0, label="Landing limit (+-5 deg)")
    axs[3].axhline(-5.0, color="#ef4444", linestyle="--", linewidth=1.0)
    axs[3].set_ylabel("Tilt (deg)", color="#f8fafc", fontsize=10)
    axs[3].set_xlabel("Time (s)", color="#f8fafc", fontsize=10)
    axs[3].legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved trajectory plot to {output_path}")


if __name__ == "__main__":
    run_random_rollout()
