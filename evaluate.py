"""Benchmark and evaluation script comparing PID, PPO (Nominal), and PPO (Randomized).

Sweeps wind disturbances and sensor observation noise across identical fixed test seeds,
and outputs comparative CSV metrics, Markdown tables, and sensitivity plots.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import glob
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from controllers.pid import HoverSlamPID
from envs.landing_env import HoverSlamEnv
from sim.rocket import IDX_VY


SWEEP_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def resolve_model_path(path_spec: str) -> Path | None:
    """Find a valid model zip file from a file path, directory, or run name."""
    p = Path(path_spec)
    if p.is_file() and p.suffix == ".zip":
        return p

    if p.is_dir():
        final_candidate = p / "final_model.zip"
        if final_candidate.is_file():
            return final_candidate
        zips = sorted(glob.glob(str(p / "*.zip")))
        if zips:
            return Path(zips[-1])

    # Search inside models/
    candidate_dir = REPO_ROOT / "models" / path_spec
    if candidate_dir.is_dir():
        final_candidate = candidate_dir / "final_model.zip"
        if final_candidate.is_file():
            return final_candidate
        zips = sorted(glob.glob(str(candidate_dir / "*.zip")))
        if zips:
            return Path(zips[-1])

    return None


def run_single_condition(
    controller_type: str,
    controller_obj: Any,
    num_episodes: int,
    base_seed: int,
    wind_strength: float,
    sensor_noise: float,
) -> dict[str, float]:
    """Run evaluation episodes on HoverSlamEnv under a specific disturbance condition."""
    env = HoverSlamEnv(
        vertical_only=False,
        wind_strength=wind_strength,
        sensor_noise=sensor_noise,
        mass_randomization=0.0,
        randomize_start=True,
    )

    landed_count = 0
    touchdown_speeds: list[float] = []
    pad_distances: list[float] = []
    fuel_remaining: list[float] = []

    for i in range(num_episodes):
        test_seed = base_seed + i
        obs, info = env.reset(seed=test_seed)

        if controller_type == "pid":
            controller_obj.reset()

        while True:
            if controller_type == "pid":
                action = controller_obj.compute_action(obs)
            else:
                action, _ = controller_obj.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if info["landed"]:
            landed_count += 1

        if info.get("touchdown_vy") is not None:
            td_speed = abs(float(info["touchdown_vy"]))
        else:
            td_speed = abs(float(env.state[IDX_VY]))
        touchdown_speeds.append(td_speed)

        pad_distances.append(float(info["distance_from_pad"]))
        fuel_remaining.append(float(info["fuel_remaining"]))

    env.close()

    success_rate = (landed_count / num_episodes) * 100.0
    mean_speed = float(np.mean(touchdown_speeds)) if touchdown_speeds else 0.0
    mean_dist = float(np.mean(pad_distances)) if pad_distances else 0.0
    mean_fuel = float(np.mean(fuel_remaining)) if fuel_remaining else 0.0

    return {
        "success_rate": success_rate,
        "mean_touchdown_speed": mean_speed,
        "mean_distance_from_pad": mean_dist,
        "mean_fuel_remaining": mean_fuel,
        "successful_landings": landed_count,
        "total_episodes": num_episodes,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI options for benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="Fair benchmark of PID vs. PPO Nominal vs. PPO Randomized under identical test scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quick",
        action="store_true",
        help="Run fast validation benchmark (100 episodes per condition).",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Run comprehensive full benchmark (1000 episodes per condition).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Explicitly override number of episodes per condition.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=10000,
        help="Base random seed for test scenarios (seeds base_seed to base_seed + episodes - 1).",
    )
    parser.add_argument(
        "--nominal-models",
        nargs="+",
        default=[
            "models/ppo_nominal_s0/final_model.zip",
            "models/ppo_nominal_s1/final_model.zip",
            "models/ppo_nominal_s2/final_model.zip",
        ],
        help="Model paths or run names for PPO Nominal seeds 0, 1, 2.",
    )
    parser.add_argument(
        "--randomized-models",
        nargs="+",
        default=[
            "models/ppo_randomized_s0/final_model.zip",
            "models/ppo_randomized_s1/final_model.zip",
            "models/ppo_randomized_s2/final_model.zip",
        ],
        help="Model paths or run names for PPO Randomized seeds 0, 1, 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save benchmark.csv, benchmark_table.md, and plots.",
    )

    args = parser.parse_args()

    # Determine episode count
    if args.episodes is not None:
        args.num_episodes = args.episodes
    elif args.full:
        args.num_episodes = 1000
    else:
        # Default or --quick
        args.num_episodes = 100

    return args


def plot_sensitivity(
    x_vals: list[float],
    pid_scores: list[float],
    ppo_nom_means: list[float],
    ppo_nom_mins: list[float],
    ppo_nom_maxs: list[float],
    ppo_rand_means: list[float],
    ppo_rand_mins: list[float],
    ppo_rand_maxs: list[float],
    param_name: str,
    param_label: str,
    base_seed: int,
    num_episodes: int,
    out_path: Path,
) -> None:
    """Generate aesthetic dark-mode sensitivity curve with shaded min-max error bands for PPO."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)

    # Dark aerospace theme styling
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.25, color="#475569")
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.0)

    # 1. Classical PID (Cyan / Sky)
    ax.plot(
        x_vals,
        pid_scores,
        color="#38bdf8",
        marker="o",
        markersize=6,
        linewidth=2.2,
        label="Classical Cascaded PID",
        zorder=4,
    )

    # 2. PPO Nominal Policy (Amber / Orange)
    ax.plot(
        x_vals,
        ppo_nom_means,
        color="#f59e0b",
        marker="s",
        markersize=5,
        linewidth=2.2,
        label="PPO Nominal (Mean across 3 seeds)",
        zorder=3,
    )
    ax.fill_between(
        x_vals,
        ppo_nom_mins,
        ppo_nom_maxs,
        color="#f59e0b",
        alpha=0.22,
        label="PPO Nominal (Seed Min-Max Band)",
        zorder=2,
    )

    # 3. PPO Randomized Policy (Emerald / Green)
    ax.plot(
        x_vals,
        ppo_rand_means,
        color="#10b981",
        marker="^",
        markersize=6,
        linewidth=2.2,
        label="PPO Randomized (Mean across 3 seeds)",
        zorder=3,
    )
    ax.fill_between(
        x_vals,
        ppo_rand_mins,
        ppo_rand_maxs,
        color="#10b981",
        alpha=0.22,
        label="PPO Randomized (Seed Min-Max Band)",
        zorder=2,
    )

    ax.set_title(
        f"Landing Success Rate vs. {param_name}\n"
        f"Benchmark on Fixed Test Seeds ({base_seed}–{base_seed + num_episodes - 1})",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(param_label, color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_ylabel("Landing Success Rate (%)", color="#e2e8f0", fontsize=11, labelpad=8)
    ax.set_ylim(-5.0, 105.0)
    ax.set_xticks(x_vals)

    ax.legend(
        loc="lower left",
        facecolor="#1e293b",
        edgecolor="#475569",
        labelcolor="#f8fafc",
        fontsize=9,
        framealpha=0.95,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[Plot Saved] {out_path}")


def main() -> None:
    """Execute complete benchmarking suite and generate outputs."""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("               HOVERSLAM CONTROLLER BENCHMARK")
    print("=" * 72)
    print(f"Mode:                      {'Quick Test' if args.num_episodes < 1000 else 'Full Benchmark'}")
    print(f"Episodes Per Condition:    {args.num_episodes:,}")
    print(f"Test Seed Range:           {args.base_seed} to {args.base_seed + args.num_episodes - 1}")
    print(f"Wind Sweep Points:         {SWEEP_POINTS}")
    print(f"Sensor Noise Sweep Points: {SWEEP_POINTS}")
    print("=" * 72 + "\n")

    # Initialize PID controller
    sample_env = HoverSlamEnv()
    pid_controller = HoverSlamPID(dt=sample_env.dt)
    sample_env.close()

    # Load PPO models
    ppo_nom_models: list[tuple[int, PPO, str]] = []
    for idx, spec in enumerate(args.nominal_models):
        mpath = resolve_model_path(spec)
        if mpath is None:
            raise FileNotFoundError(f"Nominal model not found: {spec}")
        print(f"Loaded Nominal Seed {idx} model from: {mpath}")
        ppo_nom_models.append((idx, PPO.load(str(mpath)), str(mpath)))

    ppo_rand_models: list[tuple[int, PPO, str]] = []
    for idx, spec in enumerate(args.randomized_models):
        mpath = resolve_model_path(spec)
        if mpath is None:
            raise FileNotFoundError(f"Randomized model not found: {spec}")
        print(f"Loaded Randomized Seed {idx} model from: {mpath}")
        ppo_rand_models.append((idx, PPO.load(str(mpath)), str(mpath)))

    print("\nStarting evaluation sweeps across identical test scenarios...\n")
    t_start = time.time()

    # Data structures for results
    csv_rows: list[dict[str, Any]] = []

    # Cache for evaluated conditions: key = (controller_id, wind, noise)
    eval_cache: dict[tuple[str, float, float], dict[str, float]] = {}

    def get_eval(ctrl_id: str, ctrl_type: str, ctrl_obj: Any, w: float, n: float) -> dict[str, float]:
        key = (ctrl_id, w, n)
        if key not in eval_cache:
            res = run_single_condition(
                controller_type=ctrl_type,
                controller_obj=ctrl_obj,
                num_episodes=args.num_episodes,
                base_seed=args.base_seed,
                wind_strength=w,
                sensor_noise=n,
            )
            eval_cache[key] = res
        return eval_cache[key]

    # -------------------------------------------------------------------------
    # 1. Sweep Wind (noise = 0.0)
    # -------------------------------------------------------------------------
    print(">>> Executing Wind Sweep (wind_strength in [0.0, 1.0], noise = 0.0)...")
    wind_pid_scores: list[float] = []
    wind_nom_means: list[float] = []
    wind_nom_mins: list[float] = []
    wind_nom_maxs: list[float] = []
    wind_rand_means: list[float] = []
    wind_rand_mins: list[float] = []
    wind_rand_maxs: list[float] = []

    wind_table_data: list[dict[str, Any]] = []

    for w in SWEEP_POINTS:
        # Evaluate PID
        pid_res = get_eval("PID", "pid", pid_controller, w, 0.0)
        wind_pid_scores.append(pid_res["success_rate"])
        csv_rows.append({
            "sweep_type": "wind",
            "parameter_value": w,
            "controller": "PID",
            "seed": "N/A",
            "success_rate": pid_res["success_rate"],
            "mean_touchdown_speed": pid_res["mean_touchdown_speed"],
            "mean_distance_from_pad": pid_res["mean_distance_from_pad"],
            "mean_fuel_remaining": pid_res["mean_fuel_remaining"],
            "num_episodes": args.num_episodes,
        })

        # Evaluate PPO Nominal (3 seeds)
        nom_seed_scores: list[float] = []
        nom_seed_speeds: list[float] = []
        nom_seed_dists: list[float] = []
        nom_seed_fuels: list[float] = []

        for s_idx, m_obj, _ in ppo_nom_models:
            n_res = get_eval(f"PPO_Nom_s{s_idx}", "ppo", m_obj, w, 0.0)
            nom_seed_scores.append(n_res["success_rate"])
            nom_seed_speeds.append(n_res["mean_touchdown_speed"])
            nom_seed_dists.append(n_res["mean_distance_from_pad"])
            nom_seed_fuels.append(n_res["mean_fuel_remaining"])
            csv_rows.append({
                "sweep_type": "wind",
                "parameter_value": w,
                "controller": "PPO_Nominal",
                "seed": s_idx,
                "success_rate": n_res["success_rate"],
                "mean_touchdown_speed": n_res["mean_touchdown_speed"],
                "mean_distance_from_pad": n_res["mean_distance_from_pad"],
                "mean_fuel_remaining": n_res["mean_fuel_remaining"],
                "num_episodes": args.num_episodes,
            })

        nom_mean = float(np.mean(nom_seed_scores))
        nom_min = float(np.min(nom_seed_scores))
        nom_max = float(np.max(nom_seed_scores))
        wind_nom_means.append(nom_mean)
        wind_nom_mins.append(nom_min)
        wind_nom_maxs.append(nom_max)

        # Evaluate PPO Randomized (3 seeds)
        rand_seed_scores: list[float] = []
        rand_seed_speeds: list[float] = []
        rand_seed_dists: list[float] = []
        rand_seed_fuels: list[float] = []

        for s_idx, m_obj, _ in ppo_rand_models:
            r_res = get_eval(f"PPO_Rand_s{s_idx}", "ppo", m_obj, w, 0.0)
            rand_seed_scores.append(r_res["success_rate"])
            rand_seed_speeds.append(r_res["mean_touchdown_speed"])
            rand_seed_dists.append(r_res["mean_distance_from_pad"])
            rand_seed_fuels.append(r_res["mean_fuel_remaining"])
            csv_rows.append({
                "sweep_type": "wind",
                "parameter_value": w,
                "controller": "PPO_Randomized",
                "seed": s_idx,
                "success_rate": r_res["success_rate"],
                "mean_touchdown_speed": r_res["mean_touchdown_speed"],
                "mean_distance_from_pad": r_res["mean_distance_from_pad"],
                "mean_fuel_remaining": r_res["mean_fuel_remaining"],
                "num_episodes": args.num_episodes,
            })

        rand_mean = float(np.mean(rand_seed_scores))
        rand_min = float(np.min(rand_seed_scores))
        rand_max = float(np.max(rand_seed_scores))
        wind_rand_means.append(rand_mean)
        wind_rand_mins.append(rand_min)
        wind_rand_maxs.append(rand_max)

        wind_table_data.append({
            "param": w,
            "pid_succ": pid_res["success_rate"],
            "pid_speed": pid_res["mean_touchdown_speed"],
            "pid_fuel": pid_res["mean_fuel_remaining"],
            "nom_succ_mean": nom_mean,
            "nom_succ_min": nom_min,
            "nom_succ_max": nom_max,
            "nom_speed": float(np.mean(nom_seed_speeds)),
            "rand_succ_mean": rand_mean,
            "rand_succ_min": rand_min,
            "rand_succ_max": rand_max,
            "rand_speed": float(np.mean(rand_seed_speeds)),
        })

        print(
            f"  Wind {w:4.2f} | PID: {pid_res['success_rate']:5.1f}% | "
            f"PPO Nom: {nom_mean:5.1f}% [{nom_min:.0f}-{nom_max:.0f}%] | "
            f"PPO Rand: {rand_mean:5.1f}% [{rand_min:.0f}-{rand_max:.0f}%]"
        )

    # -------------------------------------------------------------------------
    # 2. Sweep Sensor Noise (wind = 0.0)
    # -------------------------------------------------------------------------
    print("\n>>> Executing Sensor Noise Sweep (sensor_noise in [0.0, 1.0], wind = 0.0)...")
    noise_pid_scores: list[float] = []
    noise_nom_means: list[float] = []
    noise_nom_mins: list[float] = []
    noise_nom_maxs: list[float] = []
    noise_rand_means: list[float] = []
    noise_rand_mins: list[float] = []
    noise_rand_maxs: list[float] = []

    noise_table_data: list[dict[str, Any]] = []

    for n in SWEEP_POINTS:
        # Evaluate PID
        pid_res = get_eval("PID", "pid", pid_controller, 0.0, n)
        noise_pid_scores.append(pid_res["success_rate"])
        csv_rows.append({
            "sweep_type": "noise",
            "parameter_value": n,
            "controller": "PID",
            "seed": "N/A",
            "success_rate": pid_res["success_rate"],
            "mean_touchdown_speed": pid_res["mean_touchdown_speed"],
            "mean_distance_from_pad": pid_res["mean_distance_from_pad"],
            "mean_fuel_remaining": pid_res["mean_fuel_remaining"],
            "num_episodes": args.num_episodes,
        })

        # Evaluate PPO Nominal (3 seeds)
        nom_seed_scores = []
        nom_seed_speeds = []
        nom_seed_dists = []
        nom_seed_fuels = []

        for s_idx, m_obj, _ in ppo_nom_models:
            n_res = get_eval(f"PPO_Nom_s{s_idx}", "ppo", m_obj, 0.0, n)
            nom_seed_scores.append(n_res["success_rate"])
            nom_seed_speeds.append(n_res["mean_touchdown_speed"])
            nom_seed_dists.append(n_res["mean_distance_from_pad"])
            nom_seed_fuels.append(n_res["mean_fuel_remaining"])
            csv_rows.append({
                "sweep_type": "noise",
                "parameter_value": n,
                "controller": "PPO_Nominal",
                "seed": s_idx,
                "success_rate": n_res["success_rate"],
                "mean_touchdown_speed": n_res["mean_touchdown_speed"],
                "mean_distance_from_pad": n_res["mean_distance_from_pad"],
                "mean_fuel_remaining": n_res["mean_fuel_remaining"],
                "num_episodes": args.num_episodes,
            })

        nom_mean = float(np.mean(nom_seed_scores))
        nom_min = float(np.min(nom_seed_scores))
        nom_max = float(np.max(nom_seed_scores))
        noise_nom_means.append(nom_mean)
        noise_nom_mins.append(nom_min)
        noise_nom_maxs.append(nom_max)

        # Evaluate PPO Randomized (3 seeds)
        rand_seed_scores = []
        rand_seed_speeds = []
        rand_seed_dists = []
        rand_seed_fuels = []

        for s_idx, m_obj, _ in ppo_rand_models:
            r_res = get_eval(f"PPO_Rand_s{s_idx}", "ppo", m_obj, 0.0, n)
            rand_seed_scores.append(r_res["success_rate"])
            rand_seed_speeds.append(r_res["mean_touchdown_speed"])
            rand_seed_dists.append(r_res["mean_distance_from_pad"])
            rand_seed_fuels.append(r_res["mean_fuel_remaining"])
            csv_rows.append({
                "sweep_type": "noise",
                "parameter_value": n,
                "controller": "PPO_Randomized",
                "seed": s_idx,
                "success_rate": r_res["success_rate"],
                "mean_touchdown_speed": r_res["mean_touchdown_speed"],
                "mean_distance_from_pad": r_res["mean_distance_from_pad"],
                "mean_fuel_remaining": r_res["mean_fuel_remaining"],
                "num_episodes": args.num_episodes,
            })

        rand_mean = float(np.mean(rand_seed_scores))
        rand_min = float(np.min(rand_seed_scores))
        rand_max = float(np.max(rand_seed_scores))
        noise_rand_means.append(rand_mean)
        noise_rand_mins.append(rand_min)
        noise_rand_maxs.append(rand_max)

        noise_table_data.append({
            "param": n,
            "pid_succ": pid_res["success_rate"],
            "pid_speed": pid_res["mean_touchdown_speed"],
            "pid_fuel": pid_res["mean_fuel_remaining"],
            "nom_succ_mean": nom_mean,
            "nom_succ_min": nom_min,
            "nom_succ_max": nom_max,
            "nom_speed": float(np.mean(nom_seed_speeds)),
            "rand_succ_mean": rand_mean,
            "rand_succ_min": rand_min,
            "rand_succ_max": rand_max,
            "rand_speed": float(np.mean(rand_seed_speeds)),
        })

        print(
            f"  Noise {n:4.2f} | PID: {pid_res['success_rate']:5.1f}% | "
            f"PPO Nom: {nom_mean:5.1f}% [{nom_min:.0f}-{nom_max:.0f}%] | "
            f"PPO Rand: {rand_mean:5.1f}% [{rand_min:.0f}-{rand_max:.0f}%]"
        )

    t_elapsed = time.time() - t_start
    print(f"\nAll evaluations completed in {t_elapsed:.1f} seconds.")

    # -------------------------------------------------------------------------
    # 3. Write results/benchmark.csv
    # -------------------------------------------------------------------------
    csv_path = out_dir / "benchmark.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sweep_type",
            "parameter_value",
            "controller",
            "seed",
            "success_rate",
            "mean_touchdown_speed",
            "mean_distance_from_pad",
            "mean_fuel_remaining",
            "num_episodes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
    print(f"[CSV Saved] {csv_path}")

    # -------------------------------------------------------------------------
    # 4. Generate Sensitivity Plots
    # -------------------------------------------------------------------------
    wind_plot_path = out_dir / "success_vs_wind.png"
    plot_sensitivity(
        x_vals=SWEEP_POINTS,
        pid_scores=wind_pid_scores,
        ppo_nom_means=wind_nom_means,
        ppo_nom_mins=wind_nom_mins,
        ppo_nom_maxs=wind_nom_maxs,
        ppo_rand_means=wind_rand_means,
        ppo_rand_mins=wind_rand_mins,
        ppo_rand_maxs=wind_rand_maxs,
        param_name="Wind Disturbance Strength",
        param_label="Wind Strength [0.0 - 1.0] (0 to ~8 m/s peak equivalent force)",
        base_seed=args.base_seed,
        num_episodes=args.num_episodes,
        out_path=wind_plot_path,
    )

    noise_plot_path = out_dir / "success_vs_noise.png"
    plot_sensitivity(
        x_vals=SWEEP_POINTS,
        pid_scores=noise_pid_scores,
        ppo_nom_means=noise_nom_means,
        ppo_nom_mins=noise_nom_mins,
        ppo_nom_maxs=noise_nom_maxs,
        ppo_rand_means=noise_rand_means,
        ppo_rand_mins=noise_rand_mins,
        ppo_rand_maxs=noise_rand_maxs,
        param_name="Sensor Observation Noise",
        param_label="Sensor Noise Scale [0.0 - 1.0] (Multipliers on signal standard deviations)",
        base_seed=args.base_seed,
        num_episodes=args.num_episodes,
        out_path=noise_plot_path,
    )

    # -------------------------------------------------------------------------
    # 5. Write results/benchmark_table.md
    # -------------------------------------------------------------------------
    table_path = out_dir / "benchmark_table.md"
    with open(table_path, mode="w", encoding="utf-8") as f:
        f.write("# HoverSlam Controller Benchmark Report\n\n")
        f.write(f"- **Benchmark Protocol:** Fixed evaluation seeds (`{args.base_seed}`–`{args.base_seed + args.num_episodes - 1}`)\n")
        f.write(f"- **Episodes Per Condition:** {args.num_episodes:,}\n")
        f.write(f"- **Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. Wind Disturbance Sensitivity Sweep (Noise = 0.0)\n\n")
        f.write("| Wind Strength | PID Success (%) | PPO Nominal Mean [Min-Max] (%) | PPO Randomized Mean [Min-Max] (%) | PID TD Speed (m/s) | PID Fuel (kg) |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for d in wind_table_data:
            f.write(
                f"| {d['param']:.2f} | {d['pid_succ']:5.1f}% | "
                f"{d['nom_succ_mean']:5.1f}% [{d['nom_succ_min']:.0f}-{d['nom_succ_max']:.0f}%] | "
                f"{d['rand_succ_mean']:5.1f}% [{d['rand_succ_min']:.0f}-{d['rand_succ_max']:.0f}%] | "
                f"{d['pid_speed']:5.2f} | {d['pid_fuel']:5.1f} |\n"
            )

        f.write("\n## 2. Sensor Observation Noise Sensitivity Sweep (Wind = 0.0)\n\n")
        f.write("| Sensor Noise | PID Success (%) | PPO Nominal Mean [Min-Max] (%) | PPO Randomized Mean [Min-Max] (%) | PID TD Speed (m/s) | PID Fuel (kg) |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for d in noise_table_data:
            f.write(
                f"| {d['param']:.2f} | {d['pid_succ']:5.1f}% | "
                f"{d['nom_succ_mean']:5.1f}% [{d['nom_succ_min']:.0f}-{d['nom_succ_max']:.0f}%] | "
                f"{d['rand_succ_mean']:5.1f}% [{d['rand_succ_min']:.0f}-{d['rand_succ_max']:.0f}%] | "
                f"{d['pid_speed']:5.2f} | {d['pid_fuel']:5.1f} |\n"
            )

        f.write("\n## 3. Visualizations\n\n")
        f.write("- **Wind Sweep Plot:** ![Wind Sensitivity](success_vs_wind.png)\n")
        f.write("- **Sensor Noise Sweep Plot:** ![Noise Sensitivity](success_vs_noise.png)\n")

    print(f"[Markdown Table Saved] {table_path}\n")

    print("=" * 72)
    print("                 BENCHMARK EXECUTION COMPLETED")
    print("=" * 72)
    print(f"Generated Files in '{out_dir}':")
    print(f" - CSV Dataset:     {csv_path}")
    print(f" - Markdown Table:  {table_path}")
    print(f" - Wind Plot:       {wind_plot_path}")
    print(f" - Noise Plot:      {noise_plot_path}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
