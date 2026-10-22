"""Training script for HoverSlam RL agent using Stable-Baselines3 PPO."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any, Callable

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from envs.landing_env import HoverSlamEnv


class FlagOrValueAction(argparse.Action):
    """Argparse action that supports both boolean flags and explicit values.

    Examples:
        --wind           -> True (uses default active float)
        --wind 250.0     -> 250.0
        (omitted)        -> False
    """

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        nargs: str | int | None = None,
        const: Any = True,
        default: Any = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs="?",
            const=const,
            default=default,
            **kwargs,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        if values is None or values is True:
            setattr(namespace, self.dest, True)
        elif isinstance(values, str) and values.lower() in ("true", "yes", "1"):
            setattr(namespace, self.dest, True)
        elif isinstance(values, str) and values.lower() in ("false", "no", "0"):
            setattr(namespace, self.dest, False)
        else:
            try:
                setattr(namespace, self.dest, float(values))
            except ValueError:
                setattr(namespace, self.dest, values)


class LandingStatsCallback(BaseCallback):
    """Custom callback logging landing success rate and touchdown speed to TensorBoard.

    Maintains a rolling window of the last 100 episodes from info['landed']
    and records:
      - landing/success_rate: Fraction of soft landings in [0.0, 1.0]
      - landing/mean_touchdown_speed: Mean vertical impact speed in m/s
    """

    def __init__(self, window_size: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window_size = int(window_size)
        self.landed_history: deque[float] = deque(maxlen=self.window_size)
        self.touchdown_speed_history: deque[float] = deque(maxlen=self.window_size)
        self.total_episodes: int = 0
        self.total_landings: int = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if done:
                    self.total_episodes += 1
                    term_info = info.get("terminal_info", info)

                    is_landed = bool(term_info.get("landed", False))
                    self.landed_history.append(1.0 if is_landed else 0.0)
                    if is_landed:
                        self.total_landings += 1

                    touchdown_vy = term_info.get("touchdown_vy")
                    if touchdown_vy is not None:
                        self.touchdown_speed_history.append(abs(float(touchdown_vy)))

        # Record metrics if rolling statistics are available
        if len(self.landed_history) > 0:
            success_rate = float(np.mean(self.landed_history))
            self.logger.record("landing/success_rate", success_rate)
            self.logger.record("rollout/landing_success_rate", success_rate)

        if len(self.touchdown_speed_history) > 0:
            mean_speed = float(np.mean(self.touchdown_speed_history))
            self.logger.record("landing/mean_touchdown_speed", mean_speed)
            self.logger.record("rollout/mean_touchdown_speed", mean_speed)

        return True

    def _on_rollout_end(self) -> None:
        """Ensure latest rolling metrics are refreshed before logger dumps to disk."""
        if len(self.landed_history) > 0:
            success_rate = float(np.mean(self.landed_history))
            self.logger.record("landing/success_rate", success_rate)
            self.logger.record("rollout/landing_success_rate", success_rate)

        if len(self.touchdown_speed_history) > 0:
            mean_speed = float(np.mean(self.touchdown_speed_history))
            self.logger.record("landing/mean_touchdown_speed", mean_speed)
            self.logger.record("rollout/mean_touchdown_speed", mean_speed)


def make_env(
    rank: int,
    seed: int,
    vertical_only: bool = False,
    wind_strength: float = 0.0,
    sensor_noise: float = 0.0,
    mass_randomization: float = 0.0,
    randomize_start: bool = False,
) -> Callable[[], gym.Env]:
    """Create an isolated, seeded factory for HoverSlamEnv."""

    def _init() -> gym.Env:
        env = HoverSlamEnv(
            vertical_only=vertical_only,
            wind_strength=wind_strength,
            sensor_noise=sensor_noise,
            mass_randomization=mass_randomization,
            randomize_start=randomize_start,
        )
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train HoverSlam rocket landing agent using Stable-Baselines3 PPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment and policy.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1_000_000,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel vectorized simulation environments.",
    )
    parser.add_argument(
        "--vertical-only",
        action="store_true",
        help="Lock pitch and lateral motion to 1D vertical descent.",
    )
    parser.add_argument(
        "--wind",
        action=FlagOrValueAction,
        help="Enable wind disturbance (flag sets 500.0 N, or specify float).",
    )
    parser.add_argument(
        "--noise",
        action=FlagOrValueAction,
        help="Enable sensor observation noise (flag sets 0.02 std, or specify float).",
    )
    parser.add_argument(
        "--randomize",
        action=FlagOrValueAction,
        help="Enable domain randomization (randomized initial state and vehicle mass).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Identifier for TensorBoard runs/<run-name> and models/<run-name>/.",
    )

    return parser.parse_args(args)


def train(args: argparse.Namespace) -> tuple[PPO, dict[str, Any]]:
    """Execute PPO training pipeline according to CLI configuration."""
    # Seed all global RNGs
    set_random_seed(args.seed)

    # Resolve physical disturbance & randomization parameters
    if isinstance(args.wind, bool):
        wind_strength = 500.0 if args.wind else 0.0
    else:
        wind_strength = float(args.wind)

    if isinstance(args.noise, bool):
        sensor_noise = 0.02 if args.noise else 0.0
    else:
        sensor_noise = float(args.noise)

    if isinstance(args.randomize, bool):
        randomize_start = args.randomize
        mass_randomization = 0.10 if args.randomize else 0.0
    else:
        randomize_start = True
        mass_randomization = float(args.randomize)

    # Determine unique run identifier
    if args.run_name:
        run_name = str(args.run_name).strip()
    else:
        mode_str = "vert" if args.vertical_only else "planar"
        cond_parts = []
        if wind_strength > 0.0:
            cond_parts.append("wind")
        if sensor_noise > 0.0:
            cond_parts.append("noise")
        if randomize_start or mass_randomization > 0.0:
            cond_parts.append("dr")
        cond_str = f"_{'_'.join(cond_parts)}" if cond_parts else "_nominal"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_{mode_str}{cond_str}_s{args.seed}_{timestamp}"

    # Setup directories
    runs_dir = REPO_ROOT / "runs" / run_name
    models_dir = REPO_ROOT / "models" / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("                    HOVERSLAM PPO TRAINING")
    print("=" * 70)
    print(f"Run Name:                  {run_name}")
    print(f"Random Seed:               {args.seed}")
    print(f"Total Timesteps:           {args.total_steps:,}")
    print(f"Parallel Environments:     {args.n_envs}")
    print(f"Vertical Only Mode:        {'Enabled' if args.vertical_only else 'Disabled (Full 2D)'}")
    print(f"Wind Disturbance:          {wind_strength:.1f} N ({'Enabled' if wind_strength > 0 else 'Disabled'})")
    print(f"Sensor Noise:              {sensor_noise:.4f} ({'Enabled' if sensor_noise > 0 else 'Disabled'})")
    print(f"Domain Randomization:      {'Enabled' if randomize_start else 'Disabled'} (mass_rand={mass_randomization:.2f})")
    print(f"TensorBoard Log Dir:       {runs_dir}")
    print(f"Model Checkpoint Dir:      {models_dir}")
    print("=" * 70 + "\n")

    # Instantiate vectorized environments
    env_fns = [
        make_env(
            rank=i,
            seed=args.seed,
            vertical_only=args.vertical_only,
            wind_strength=wind_strength,
            sensor_noise=sensor_noise,
            mass_randomization=mass_randomization,
            randomize_start=randomize_start,
        )
        for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    try:
        # PPO Agent with MlpPolicy and net_arch [64, 64], default hyperparameters otherwise
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=dict(net_arch=[64, 64]),
            seed=args.seed,
            verbose=1,
        )

        # Configure TensorBoard logger to write directly to runs/<run-name>
        tb_logger = configure(str(runs_dir), ["stdout", "tensorboard"])
        model.set_logger(tb_logger)

        # Callbacks
        landing_cb = LandingStatsCallback(window_size=100)

        # Checkpoint every 100k timesteps (save_freq in vec_env steps = 100k // n_envs)
        checkpoint_freq_env_steps = max(1, 100_000 // args.n_envs)
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq_env_steps,
            save_path=str(models_dir),
            name_prefix="ppo_model",
            verbose=1,
        )

        callback_list = CallbackList([landing_cb, checkpoint_cb])

        # Execute training loop
        model.learn(
            total_timesteps=args.total_steps,
            callback=callback_list,
            reset_num_timesteps=True,
        )

        # Save final model
        final_model_path = models_dir / "final_model.zip"
        model.save(str(final_model_path))
        print(f"\n[Saved] Final trained model saved to: {final_model_path}")

    finally:
        vec_env.close()

    # Calculate final summary metrics
    completed_episodes = landing_cb.total_episodes
    successful_landings = landing_cb.total_landings
    if len(landing_cb.landed_history) > 0:
        recent_window = len(landing_cb.landed_history)
        recent_success_rate = float(np.mean(landing_cb.landed_history)) * 100.0
    else:
        recent_window = 0
        recent_success_rate = 0.0

    if len(landing_cb.touchdown_speed_history) > 0:
        recent_speed_window = len(landing_cb.touchdown_speed_history)
        recent_mean_speed = float(np.mean(landing_cb.touchdown_speed_history))
    else:
        recent_speed_window = 0
        recent_mean_speed = 0.0

    # Print clear, comprehensive summary
    print("\n" + "=" * 70)
    print("                   HOVERSLAM PPO TRAINING COMPLETE")
    print("=" * 70)
    print(f"Run Name:                  {run_name}")
    print(f"Total Steps Trained:       {model.num_timesteps:,} / {args.total_steps:,}")
    print(f"Parallel Environments:     {args.n_envs}")
    print(f"Configuration:             {'1D Vertical' if args.vertical_only else 'Full 2D Planar'}")
    print(f"Random Seed:               {args.seed}")
    print("-" * 70)
    print(f"Total Episodes Completed:  {completed_episodes:,}")
    print(f"Total Successful Landings: {successful_landings:,}")
    if recent_window > 0:
        print(f"Success Rate (last {recent_window} eps): {recent_success_rate:6.2f}%")
    else:
        print("Success Rate:              N/A (no episodes completed)")

    if recent_speed_window > 0:
        print(f"Mean Touchdown Speed:      {recent_mean_speed:6.2f} m/s (last {recent_speed_window} touchdowns)")
    else:
        print("Mean Touchdown Speed:      N/A")
    print("-" * 70)
    print(f"Final Model:               {final_model_path}")
    print(f"Checkpoints Directory:     {models_dir}")
    print(f"TensorBoard Logs:          {runs_dir}")
    print("=" * 70)
    print("\nTo inspect training curves and metrics, launch TensorBoard with:")
    print("    tensorboard --logdir runs\n")

    summary = {
        "run_name": run_name,
        "total_timesteps": model.num_timesteps,
        "total_episodes": completed_episodes,
        "total_landings": successful_landings,
        "recent_success_rate": recent_success_rate,
        "recent_mean_touchdown_speed": recent_mean_speed,
        "final_model_path": str(final_model_path),
        "models_dir": str(models_dir),
        "runs_dir": str(runs_dir),
    }

    return model, summary


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
