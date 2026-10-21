"""Evaluation and tuning script for HoverSlam PID controller."""

from __future__ import annotations

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

from controllers.pid import DEFAULT_PID_GAINS, HoverSlamPID
from envs.landing_env import HoverSlamEnv
from sim.rocket import IDX_THETA, IDX_VY, IDX_X, IDX_Y


def evaluate_pid(
    gains: dict[str, float] | None = None,
    num_episodes: int = 200,
    base_seed: int = 1000,
    plot_trajectory: bool = True,
    plot_path: str = "results/pid_trajectory.png",
) -> dict[str, float]:
    """Run evaluation episodes on HoverSlamEnv with fixed seeds and report metrics."""
    env = HoverSlamEnv(
        vertical_only=False,
        wind_strength=0.0,
        sensor_noise=0.0,
        mass_randomization=0.0,
        randomize_start=True,
    )

    controller = HoverSlamPID(gains=gains, dt=env.dt)

    successes = 0
    touchdown_speeds = []
    distances_from_pad = []
    fuels_remaining = []

    # Store trajectory data for the first episode for plotting
    sample_trajectory = None

    for i in range(num_episodes):
        seed = base_seed + i
        obs, info = env.reset(seed=seed)
        controller.reset()

        record_this = (i == 0 and plot_trajectory)
        if record_this:
            traj_t = [0.0]
            traj_y = [float(env.state[IDX_Y])]
            traj_vy = [float(env.state[IDX_VY])]
            traj_x = [float(env.state[IDX_X])]
            traj_th = [0.0]
            traj_tilt = [float(np.degrees(env.state[IDX_THETA]))]
            traj_ignited = [0]

        step_count = 0
        while True:
            action = controller.compute_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if record_this:
                traj_t.append(step_count * env.dt)
                traj_y.append(float(env.state[IDX_Y]))
                traj_vy.append(float(env.state[IDX_VY]))
                traj_x.append(float(env.state[IDX_X]))
                traj_th.append(env.last_throttle)
                traj_tilt.append(float(np.degrees(env.state[IDX_THETA])))
                traj_ignited.append(1 if controller.ignited else 0)

            if terminated or truncated:
                break

        if info["landed"]:
            successes += 1

        if info["touchdown_vy"] is not None:
            touchdown_speeds.append(abs(info["touchdown_vy"]))
        else:
            touchdown_speeds.append(abs(float(env.state[IDX_VY])))

        distances_from_pad.append(info["distance_from_pad"])
        fuels_remaining.append(info["fuel_remaining"])

        if record_this:
            sample_trajectory = {
                "t": traj_t,
                "y": traj_y,
                "vy": traj_vy,
                "x": traj_x,
                "throttle": traj_th,
                "tilt": traj_tilt,
                "ignited": traj_ignited,
                "landed": info["landed"],
            }

    env.close()

    success_rate = (successes / num_episodes) * 100.0
    mean_speed = float(np.mean(touchdown_speeds))
    mean_dist = float(np.mean(distances_from_pad))
    mean_fuel = float(np.mean(fuels_remaining))

    results = {
        "success_rate": success_rate,
        "mean_touchdown_speed": mean_speed,
        "mean_distance_from_pad": mean_dist,
        "mean_fuel_remaining": mean_fuel,
        "successful_landings": successes,
        "total_episodes": num_episodes,
    }

    print("=" * 60)
    print(f"PID Evaluation Results ({num_episodes} nominal episodes):")
    print(f"  Success Rate:          {success_rate:6.2f}% ({successes}/{num_episodes})")
    print(f"  Mean Touchdown Speed:  {mean_speed:6.2f} m/s")
    print(f"  Mean Distance to Pad:  {mean_dist:6.2f} m")
    print(f"  Mean Fuel Remaining:   {mean_fuel:6.2f} kg")
    print("=" * 60)

    # Plot sample trajectory
    if plot_trajectory and sample_trajectory is not None:
        fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        fig.patch.set_facecolor("#0f172a")

        for ax in axs:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8")
            ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
            for spine in ax.spines.values():
                spine.set_color("#334155")

        t = sample_trajectory["t"]

        # 1. Altitude
        axs[0].plot(t, sample_trajectory["y"], color="#38bdf8", linewidth=1.8, label="Altitude y (m)")
        axs[0].axhline(0.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Ground (y=0)")
        axs[0].set_ylabel("Altitude (m)", color="#f8fafc", fontsize=10)
        axs[0].legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
        outcome_str = "SUCCESSFUL LANDING" if sample_trajectory["landed"] else "CRASH / FAILED"
        axs[0].set_title(f"HoverSlam PID Closed-Loop Landing Profile [{outcome_str}]", color="#f8fafc", fontsize=12)

        # 2. Vertical velocity
        axs[1].plot(t, sample_trajectory["vy"], color="#f59e0b", linewidth=1.8, label="vy (m/s)")
        axs[1].axhline(0.0, color="#94a3b8", linestyle=":", linewidth=1.0)
        axs[1].axhline(-2.0, color="#22c55e", linestyle="--", linewidth=1.0, label="Soft limit (-2 m/s)")
        axs[1].set_ylabel("Vertical Vel (m/s)", color="#f8fafc", fontsize=10)
        axs[1].legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

        # 3. Throttle
        axs[2].step(t, sample_trajectory["throttle"], color="#10b981", linewidth=1.8, where="post", label="Throttle Fraction")
        axs[2].set_ylabel("Throttle", color="#f8fafc", fontsize=10)
        axs[2].set_ylim(-0.05, 1.05)
        axs[2].legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

        # 4. Tilt Angle
        axs[3].plot(t, sample_trajectory["tilt"], color="#ec4899", linewidth=1.8, label="Pitch Angle (deg)")
        axs[3].axhline(0.0, color="#94a3b8", linestyle=":", linewidth=1.0)
        axs[3].axhline(5.0, color="#ef4444", linestyle="--", linewidth=1.0, label="Touchdown Limit (+-5 deg)")
        axs[3].axhline(-5.0, color="#ef4444", linestyle="--", linewidth=1.0)
        axs[3].set_ylabel("Pitch (deg)", color="#f8fafc", fontsize=10)
        axs[3].set_xlabel("Time (s)", color="#f8fafc", fontsize=10)
        axs[3].legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

        plt.tight_layout()
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"Saved PID trajectory plot to {plot_path}")

    return results


if __name__ == "__main__":
    evaluate_pid()
