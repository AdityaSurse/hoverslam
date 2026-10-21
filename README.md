# HoverSlam 🚀

HoverSlam is an autonomous 2D rocket booster landing simulation. An RL agent (PPO via Stable-Baselines3) learns to execute a hoverslam ("suicide burn") landing maneuver in a custom physics simulation, benchmarked against a classical PID controller and a domain-randomized PPO agent.

## Project Structure

```
hoverslam/
├── sim/            # Custom 2D rocket physics engine and equations of motion
├── envs/           # Gymnasium environment definitions
├── controllers/    # Classical baseline controllers (PID)
├── tests/          # Pytest unit and integration tests
├── results/        # Evaluation metrics, plots, and saved videos
├── train.py        # RL training pipeline
├── evaluate.py     # Benchmark and comparison script
├── render.py       # Trajectory visualization and rendering
├── requirements.txt # Python package dependencies
├── README.md       # Project overview
└── NOTES.md        # Technical notes and development log
```

## Quick Start

Activate the virtual environment:
```bash
source venv/bin/activate
```

Run tests:
```bash
pytest
```
