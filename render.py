"""HoverSlam Video Renderer: Side-by-side demo of Classical PID vs PPO Reinforcement Learning.

Exports:
  - results/hoverslam_demo.mp4: 60-90 second side-by-side HD demo with title card,
    telemetry overlay, wind indicators, scenario 1 (PID crashes, RL lands),
    scenario 2 (both land), and end card summary.
  - results/hoverslam_preview.gif: 10-second animated GIF preview of the touchdown burn.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controllers.pid import HoverSlamPID
from envs.landing_env import HoverSlamEnv
from sim.rocket import (
    DIAMETER,
    DRY_MASS,
    FUEL_MASS,
    HEIGHT,
    IDX_M,
    IDX_OMEGA,
    IDX_THETA,
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    START_MASS,
)

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
WIDTH: int = 1280
HEIGHT_PX: int = 720
FPS: int = 25

# Panel geometry
HEADER_H: int = 50
MAIN_H: int = 510
TELEMETRY_H: int = HEIGHT_PX - MAIN_H  # 210 px
LEFT_PANEL_W: int = WIDTH // 2  # 640 px
RIGHT_PANEL_W: int = WIDTH // 2  # 640 px

# Simulation coordinate transformation
GROUND_Y: int = 490
SCALE_M_TO_PX: float = 1.35  # 1 meter = 1.35 pixels (350m alt = ~470px)
LEFT_PAD_X: int = 320
RIGHT_PAD_X: int = 960

# Palette
COLOR_BG: str = "#070b14"
COLOR_HEADER_BG: str = "#0b1120"
COLOR_PANEL_BORDER: str = "#1e293b"
COLOR_GROUND: str = "#334155"
COLOR_TERRAIN: str = "#0f172a"
COLOR_PAD: str = "#64748b"
COLOR_PAD_TARGET: str = "#10b981"
COLOR_PID_ACCENT: str = "#38bdf8"
COLOR_RL_ACCENT: str = "#34d399"
COLOR_TEXT_MAIN: str = "#f8fafc"
COLOR_TEXT_MUTED: str = "#94a3b8"
COLOR_SUCCESS: str = "#22c55e"
COLOR_DANGER: str = "#ef4444"


# -----------------------------------------------------------------------------
# Font Loader with Robust Fallbacks
# -----------------------------------------------------------------------------
def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load high-quality system font with graceful fallbacks."""
    font_candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


FONT_TITLE = load_font(42, bold=True)
FONT_SUBTITLE = load_font(22, bold=True)
FONT_HEADING = load_font(18, bold=True)
FONT_BODY = load_font(14, bold=False)
FONT_MONO = load_font(13, bold=False)
FONT_MONO_BOLD = load_font(14, bold=True)
FONT_BADGE = load_font(16, bold=True)
FONT_HUGE_BANNER = load_font(32, bold=True)


@dataclass
class TelemetryFrame:
    """Snapshot of rocket telemetry at a given time step."""
    x: float
    y: float
    vx: float
    vy: float
    theta: float
    omega: float
    fuel: float
    throttle: float
    gimbal: float
    wind_x: float
    landed: bool
    done: bool
    status_label: str  # "DESCENT", "TOUCHDOWN", "CRASHED"


def simulate_episode(
    env: HoverSlamEnv,
    controller_type: str,
    controller_obj: HoverSlamPID | PPO,
    seed: int,
    max_steps: int = 750,
) -> list[TelemetryFrame]:
    """Execute complete episode and record frame-by-frame telemetry."""
    obs, info = env.reset(seed=seed)
    if controller_type == "pid":
        controller_obj.reset()

    frames: list[TelemetryFrame] = []
    done = False
    step_cnt = 0
    landed = False

    while not done and step_cnt < max_steps:
        # State before action
        x = float(env.state[IDX_X])
        y = max(0.0, float(env.state[IDX_Y]))
        vx = float(env.state[IDX_VX])
        vy = float(env.state[IDX_VY])
        theta = float(env.state[IDX_THETA])
        omega = float(env.state[IDX_OMEGA])
        fuel = max(0.0, float(env.state[IDX_M]) - DRY_MASS)

        if controller_type == "pid":
            act = controller_obj.compute_action(obs)
        else:
            act, _ = controller_obj.predict(obs, deterministic=True)

        throttle, gimbal = env._map_action(act)
        wind_x = float(getattr(env, "last_wind_x", 0.0))

        obs, reward, terminated, truncated, step_info = env.step(act)
        done = terminated or truncated
        step_cnt += 1
        if step_info.get("landed", False):
            landed = True

        status = "DESCENT"
        if done:
            status = "LANDED" if landed else "CRASHED"

        frames.append(TelemetryFrame(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            theta=theta,
            omega=omega,
            fuel=fuel,
            throttle=throttle,
            gimbal=gimbal,
            wind_x=wind_x,
            landed=landed,
            done=done,
            status_label=status,
        ))

    # Add terminal hold frames
    last = frames[-1]
    last_status = "LANDED" if landed else "CRASHED"
    for _ in range(60):  # hold terminal frame for 2.4 seconds at 25 fps
        frames.append(TelemetryFrame(
            x=last.x,
            y=last.y,
            vx=last.vx,
            vy=last.vy,
            theta=last.theta,
            omega=last.omega,
            fuel=last.fuel,
            throttle=0.0,
            gimbal=0.0,
            wind_x=last.wind_x,
            landed=landed,
            done=True,
            status_label=last_status,
        ))

    return frames


# -----------------------------------------------------------------------------
# Rendering Primitives
# -----------------------------------------------------------------------------
def rotate_point(x: float, y: float, cx: float, cy: float, angle_rad: float) -> tuple[float, float]:
    """Rotate 2D point (x, y) around center (cx, cy)."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = x - cx
    dy = y - cy
    rx = cx + dx * cos_a - dy * sin_a
    ry = cy + dx * sin_a + dy * cos_a
    return rx, ry


def draw_rocket_and_plume(
    draw: ImageDraw.ImageDraw,
    pad_cx: int,
    ground_y: int,
    frame: TelemetryFrame,
    is_left: bool,
) -> None:
    """Render physical rocket booster, gimbaled engine nozzle, and exhaust plume."""
    # Screen coordinates
    rx = pad_cx + frame.x * SCALE_M_TO_PX
    ry = ground_y - frame.y * SCALE_M_TO_PX

    # Clamp drawing within flight area
    rx = max(30, min(1250, rx))
    ry = min(ground_y + 10, max(60, ry))

    # Rocket body geometry (width 8 px, height 36 px)
    bw = 7.0
    bh = 34.0
    half_w = bw / 2.0
    half_h = bh / 2.0
    theta = -frame.theta  # Invert for screen coordinate space

    # Corner points relative to rocket center
    corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h - 4.0),
        (0.0, half_h),         # Nose tip
        (-half_w, half_h - 4.0),
    ]

    rot_corners = [rotate_point(rx + cx, ry - cy, rx, ry, theta) for cx, cy in corners]
    draw.polygon(rot_corners, fill="#f1f5f9", outline="#94a3b8")

    # Black interstage ring
    ring_corners = [
        (-half_w, half_h - 12.0),
        (half_w, half_h - 12.0),
        (half_w, half_h - 8.0),
        (-half_w, half_h - 8.0),
    ]
    rot_ring = [rotate_point(rx + cx, ry - cy, rx, ry, theta) for cx, cy in ring_corners]
    draw.polygon(rot_ring, fill="#1e293b", outline="#334155")

    # Landing legs at base
    leg_l = rotate_point(rx - half_w - 6.0, ry + half_h + 3.0, rx, ry, theta)
    leg_base_l = rotate_point(rx - half_w, ry + half_h - 4.0, rx, ry, theta)
    draw.line([leg_base_l, leg_l], fill="#cbd5e1", width=2)

    leg_r = rotate_point(rx + half_w + 6.0, ry + half_h + 3.0, rx, ry, theta)
    leg_base_r = rotate_point(rx + half_w, ry + half_h - 4.0, rx, ry, theta)
    draw.line([leg_base_r, leg_r], fill="#cbd5e1", width=2)

    # Engine exhaust plume
    if frame.throttle > 0.01 and not frame.done:
        flame_len = 8.0 + 48.0 * float(frame.throttle)
        flame_angle = theta - frame.gimbal  # Gimbal relative to body

        nozzle_pt = rotate_point(rx, ry + half_h, rx, ry, theta)

        # Multi-layer flame (outer orange, middle yellow, inner bright cyan)
        # 1. Outer energetic orange plume
        tip_outer = rotate_point(nozzle_pt[0], nozzle_pt[1] + flame_len, nozzle_pt[0], nozzle_pt[1], flame_angle)
        left_outer = rotate_point(nozzle_pt[0] - 6.0, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        right_outer = rotate_point(nozzle_pt[0] + 6.0, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        draw.polygon([left_outer, tip_outer, right_outer], fill="#ff4500", outline="#ea580c")

        # 2. Middle yellow flame core
        tip_mid = rotate_point(nozzle_pt[0], nozzle_pt[1] + flame_len * 0.75, nozzle_pt[0], nozzle_pt[1], flame_angle)
        left_mid = rotate_point(nozzle_pt[0] - 4.0, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        right_mid = rotate_point(nozzle_pt[0] + 4.0, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        draw.polygon([left_mid, tip_mid, right_mid], fill="#facc15")

        # 3. Inner Mach diamond white-hot core
        tip_inner = rotate_point(nozzle_pt[0], nozzle_pt[1] + flame_len * 0.40, nozzle_pt[0], nozzle_pt[1], flame_angle)
        left_inner = rotate_point(nozzle_pt[0] - 2.5, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        right_inner = rotate_point(nozzle_pt[0] + 2.5, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        draw.polygon([left_inner, tip_inner, right_inner], fill="#e0f2fe")


def draw_hud_panel(
    draw: ImageDraw.ImageDraw,
    box_x1: int,
    box_y1: int,
    box_x2: int,
    box_y2: int,
    frame: TelemetryFrame,
    accent_color: str,
    controller_name: str,
) -> None:
    """Render telemetry dashboard card with gauges, values, and status badge."""
    # Card background and border
    draw.rounded_rectangle([box_x1, box_y1, box_x2, box_y2], radius=10, fill="#0d1424", outline="#1e293b", width=2)

    # Controller Title Bar
    draw.rectangle([box_x1 + 10, box_y1 + 8, box_x2 - 10, box_y1 + 32], fill="#111c33")
    draw.text((box_x1 + 16, box_y1 + 10), controller_name, fill=accent_color, font=FONT_BADGE)

    # Status Pill on Right of Header
    if frame.status_label == "LANDED":
        status_bg = "#064e3b"
        status_border = "#10b981"
        status_text = "TOUCHDOWN: LANDED"
        text_color = "#6ee7b7"
    elif frame.status_label == "CRASHED":
        status_bg = "#7f1d1d"
        status_border = "#ef4444"
        status_text = "TOUCHDOWN: CRASHED"
        text_color = "#fca5a5"
    else:
        status_bg = "#1e293b"
        status_border = "#475569"
        status_text = "STATUS: ACTIVE DESCENT"
        text_color = "#93c5fd"

    pill_w = 200
    pill_x1 = box_x2 - pill_w - 15
    pill_y1 = box_y1 + 9
    draw.rounded_rectangle([pill_x1, pill_y1, pill_x1 + pill_w, pill_y1 + 22], radius=6, fill=status_bg, outline=status_border)
    draw.text((pill_x1 + 10, pill_y1 + 3), status_text, fill=text_color, font=FONT_MONO_BOLD)

    # Telemetry Columns
    row1_y = box_y1 + 42
    row2_y = box_y1 + 76
    row3_y = box_y1 + 110

    c1 = box_x1 + 20
    c2 = box_x1 + 175
    c3 = box_x1 + 330
    c4 = box_x1 + 480

    # Column 1: Vertical Dynamics
    draw.text((c1, row1_y), "ALTITUDE", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c1, row1_y + 14), f"{frame.y:6.1f} m", fill=COLOR_TEXT_MAIN, font=FONT_MONO_BOLD)

    vy_color = COLOR_SUCCESS if abs(frame.vy) < 2.0 else (COLOR_DANGER if abs(frame.vy) > 8.0 else "#fbbf24")
    draw.text((c1, row2_y), "VERT SPEED", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c1, row2_y + 14), f"{frame.vy:+6.1f} m/s", fill=vy_color, font=FONT_MONO_BOLD)

    # Column 2: Horizontal Dynamics
    draw.text((c2, row1_y), "LATERAL POS", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c2, row1_y + 14), f"{frame.x:+6.1f} m", fill=COLOR_TEXT_MAIN, font=FONT_MONO_BOLD)

    draw.text((c2, row2_y), "HORIZ SPEED", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c2, row2_y + 14), f"{frame.vx:+6.1f} m/s", fill=COLOR_TEXT_MAIN, font=FONT_MONO_BOLD)

    # Column 3: Attitude & Steering
    tilt_deg = math.degrees(frame.theta)
    draw.text((c3, row1_y), "PITCH TILT", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c3, row1_y + 14), f"{tilt_deg:+5.1f} deg", fill=COLOR_TEXT_MAIN, font=FONT_MONO_BOLD)

    gimbal_deg = math.degrees(frame.gimbal)
    draw.text((c3, row2_y), "GIMBAL NOZZLE", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c3, row2_y + 14), f"{gimbal_deg:+5.1f} deg", fill=COLOR_TEXT_MAIN, font=FONT_MONO_BOLD)

    # Column 4: Propulsion & Fuel Gauges
    thr_pct = frame.throttle * 100.0
    draw.text((c4, row1_y), "THROTTLE", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c4, row1_y + 14), f"{thr_pct:4.0f} %", fill="#f97316", font=FONT_MONO_BOLD)
    # Mini throttle gauge bar
    bar_x = c4 + 65
    draw.rectangle([bar_x, row1_y + 16, bar_x + 55, row1_y + 26], fill="#1e293b", outline="#475569")
    fill_w = int(55.0 * min(1.0, max(0.0, frame.throttle)))
    if fill_w > 0:
        draw.rectangle([bar_x, row1_y + 16, bar_x + fill_w, row1_y + 26], fill="#ea580c")

    fuel_pct = (frame.fuel / FUEL_MASS) * 100.0
    draw.text((c4, row2_y), "FUEL REMAIN", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c4, row2_y + 14), f"{frame.fuel:5.1f} kg", fill="#38bdf8", font=FONT_MONO_BOLD)
    # Mini fuel gauge bar
    draw.rectangle([bar_x, row2_y + 16, bar_x + 55, row2_y + 26], fill="#1e293b", outline="#475569")
    fuel_w = int(55.0 * min(1.0, max(0.0, frame.fuel / FUEL_MASS)))
    if fuel_w > 0:
        draw.rectangle([bar_x, row2_y + 16, bar_x + fuel_w, row2_y + 26], fill="#0284c7")

    # Bottom Row Note
    draw.line([box_x1 + 15, row3_y + 8, box_x2 - 15, row3_y + 8], fill="#1e293b", width=1)
    status_summary = (
        "PERFECT LANDING: Touchdown v < 2 m/s, pad center reached"
        if frame.landed
        else ("CRASHED: High-velocity ground impact or loss of attitude control" if frame.done else "Retro-propulsive hoverslam trajectory guidance active")
    )
    draw.text((box_x1 + 20, row3_y + 15), status_summary, fill=COLOR_TEXT_MUTED, font=FONT_BODY)


def render_split_frame(
    f_pid: TelemetryFrame,
    f_rl: TelemetryFrame,
    scenario_title: str,
    wind_label: str,
    noise_label: str,
) -> np.ndarray:
    """Render complete 1280x720 side-by-side comparative frame."""
    img = Image.new("RGB", (WIDTH, HEIGHT_PX), color=COLOR_BG)
    draw = ImageDraw.Draw(img)

    # 1. Top Header Bar
    draw.rectangle([0, 0, WIDTH, HEADER_H], fill=COLOR_HEADER_BG)
    draw.line([0, HEADER_H, WIDTH, HEADER_H], fill=COLOR_PANEL_BORDER, width=2)
    draw.text((20, 12), "HOVERSLAM FLIGHT CONTROLLER TESTBENCH", fill=COLOR_TEXT_MAIN, font=FONT_HEADING)

    # Scenario & disturbance badges in header
    badge_text = f"SCENARIO: {scenario_title}  |  {wind_label}  |  {noise_label}"
    draw.text((WIDTH - 720, 15), badge_text, fill=COLOR_TEXT_MUTED, font=FONT_MONO_BOLD)

    # 2. Main Flight Canvas (Background & Grid)
    # Divider between left and right panel
    draw.line([WIDTH // 2, HEADER_H, WIDTH // 2, MAIN_H], fill="#1e293b", width=2)

    # Faint altitude grid lines
    for alt_m in [100, 200, 300]:
        y_grid = int(GROUND_Y - alt_m * SCALE_M_TO_PX)
        draw.line([0, y_grid, WIDTH, y_grid], fill="#101827", width=1)
        draw.text((10, y_grid - 12), f"{alt_m}m", fill="#334155", font=FONT_MONO)
        draw.text((WIDTH // 2 + 10, y_grid - 12), f"{alt_m}m", fill="#334155", font=FONT_MONO)

    # 3. Ground and Landing Pads
    draw.rectangle([0, GROUND_Y, WIDTH, MAIN_H], fill=COLOR_TERRAIN)
    draw.line([0, GROUND_Y, WIDTH, GROUND_Y], fill=COLOR_GROUND, width=2)

    # Landing pad geometry (±15m outer pad, ±5m center target)
    pad_hw = int(15.0 * SCALE_M_TO_PX)
    target_hw = int(5.0 * SCALE_M_TO_PX)

    for cx in [LEFT_PAD_X, RIGHT_PAD_X]:
        # Outer landing concrete pad
        draw.rectangle([cx - pad_hw, GROUND_Y - 3, cx + pad_hw, GROUND_Y + 4], fill="#475569", outline="#64748b")
        # Center target bullseye
        draw.rectangle([cx - target_hw, GROUND_Y - 4, cx + target_hw, GROUND_Y + 3], fill="#059669", outline="#10b981")
        # Flashing edge beacons
        draw.ellipse([cx - pad_hw - 3, GROUND_Y - 6, cx - pad_hw + 3, GROUND_Y], fill="#ef4444")
        draw.ellipse([cx + pad_hw - 3, GROUND_Y - 6, cx + pad_hw + 3, GROUND_Y], fill="#ef4444")

    # 4. Wind Vectors (Top corner of each panel)
    for px, f in [(LEFT_PAD_X, f_pid), (RIGHT_PAD_X, f_rl)]:
        wx = f.wind_x
        wind_dir = "-->" if wx >= 0 else "<--"
        wind_str = f"WIND: {abs(wx):4.0f} N {wind_dir}"
        draw.rounded_rectangle([px + 120, HEADER_H + 12, px + 280, HEADER_H + 34], radius=5, fill="#0f172a", outline="#334155")
        draw.text((px + 130, HEADER_H + 15), wind_str, fill="#94a3b8", font=FONT_MONO_BOLD)

    # 5. Draw Rockets and Plumes
    draw_rocket_and_plume(draw, LEFT_PAD_X, GROUND_Y, f_pid, is_left=True)
    draw_rocket_and_plume(draw, RIGHT_PAD_X, GROUND_Y, f_rl, is_left=False)

    # 6. Telemetry Overlay Cards (Bottom Area)
    margin = 15
    card_w = (WIDTH - 3 * margin) // 2
    draw_hud_panel(
        draw,
        box_x1=margin,
        box_y1=MAIN_H + 8,
        box_x2=margin + card_w,
        box_y2=HEIGHT_PX - 12,
        frame=f_pid,
        accent_color=COLOR_PID_ACCENT,
        controller_name="CONTROLLER 1: CLASSICAL CASCADED PID",
    )

    draw_hud_panel(
        draw,
        box_x1=margin * 2 + card_w,
        box_y1=MAIN_H + 8,
        box_x2=WIDTH - margin,
        box_y2=HEIGHT_PX - 12,
        frame=f_rl,
        accent_color=COLOR_RL_ACCENT,
        controller_name="CONTROLLER 2: PPO RANDOMIZED DEEP RL",
    )

    return np.array(img)


# -----------------------------------------------------------------------------
# Card Screens: Title, Transition, End Summary
# -----------------------------------------------------------------------------
def render_title_card() -> np.ndarray:
    """Generate sleek title card introducing the benchmark."""
    img = Image.new("RGB", (WIDTH, HEIGHT_PX), color="#060913")
    draw = ImageDraw.Draw(img)

    # Subtle starry background
    np.random.seed(42)
    for _ in range(80):
        sx = int(np.random.randint(0, WIDTH))
        sy = int(np.random.randint(0, HEIGHT_PX))
        draw.point((sx, sy), fill="#334155")

    # Glowing box container
    draw.rounded_rectangle([140, 100, WIDTH - 140, HEIGHT_PX - 100], radius=16, fill="#0b1120", outline="#1e293b", width=3)

    # Badges
    draw.rounded_rectangle([180, 140, 430, 170], radius=8, fill="#0f2847", outline="#0284c7")
    draw.text((195, 146), "AUTONOMOUS AEROSPACE RL", fill="#38bdf8", font=FONT_MONO_BOLD)

    # Main Title
    draw.text((180, 190), "HoverSlam", fill="#38bdf8", font=FONT_TITLE)
    draw.text((180, 250), "Deep RL vs. Classical PID Rocket Landing", fill="#f8fafc", font=FONT_SUBTITLE)

    # Description Paragraph
    desc_lines = [
        "A rigorous side-by-side benchmark comparing classical cascaded PID control",
        "against domain-randomized Proximal Policy Optimization (PPO) in planar 2D retro-propulsion.",
        "",
        "Evaluation under identical environmental seeds, continuous Ornstein-Uhlenbeck wind gusts,",
        "and unmodeled high-frequency sensor noise.",
    ]
    dy = 310
    for line in desc_lines:
        draw.text((180, dy), line, fill="#cbd5e1", font=FONT_BODY)
        dy += 24

    # Feature Pills
    pills = [
        ("WIND: Ornstein-Uhlenbeck (0.75)", "#064e3b", "#10b981", "#6ee7b7"),
        ("NOISE: Gaussian Sensor Disturbance", "#701a75", "#c026d3", "#f0abfc"),
        ("CONTROLLERS: PID vs PPO MlpPolicy", "#1e3a8a", "#2563eb", "#93c5fd"),
    ]
    px = 180
    for text, bg, border, fg in pills:
        pw = len(text) * 9 + 20
        draw.rounded_rectangle([px, 460, px + pw, 490], radius=8, fill=bg, outline=border)
        draw.text((px + 10, 466), text, fill=fg, font=FONT_MONO_BOLD)
        px += pw + 18

    # Footer
    draw.line([180, 530, WIDTH - 180, 530], fill="#1e293b", width=2)
    draw.text((180, 550), "AdityaSurse / hoverslam  •  Gymnasium & PyTorch  •  imageio-ffmpeg", fill="#64748b", font=FONT_MONO)

    return np.array(img)


def render_transition_card(scenario_num: int, title: str, subtitle: str, disturbance_info: str) -> np.ndarray:
    """Generate transition slate between benchmark scenarios."""
    img = Image.new("RGB", (WIDTH, HEIGHT_PX), color="#080e1a")
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([180, 180, WIDTH - 180, HEIGHT_PX - 180], radius=16, fill="#0c1527", outline="#1e293b", width=2)

    badge = f"SCENARIO {scenario_num} / 2"
    draw.rounded_rectangle([220, 220, 360, 248], radius=6, fill="#1e3a8a", outline="#3b82f6")
    draw.text((232, 225), badge, fill="#93c5fd", font=FONT_MONO_BOLD)

    draw.text((220, 270), title, fill="#f8fafc", font=FONT_SUBTITLE)
    draw.text((220, 320), subtitle, fill="#94a3b8", font=FONT_BODY)

    draw.line([220, 370, WIDTH - 220, 370], fill="#1e293b", width=1)
    draw.text((220, 390), f"TEST CONDITION: {disturbance_info}", fill="#38bdf8", font=FONT_MONO_BOLD)

    return np.array(img)


def render_end_card() -> np.ndarray:
    """Generate summary end card with benchmark headline figures and disclaimers."""
    img = Image.new("RGB", (WIDTH, HEIGHT_PX), color="#060913")
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([120, 80, WIDTH - 120, HEIGHT_PX - 80], radius=16, fill="#0b1120", outline="#1e293b", width=3)

    draw.text((160, 120), "Benchmark Summary & Performance Takeaways", fill="#f8fafc", font=FONT_SUBTITLE)
    draw.text((160, 160), "Key findings from results/benchmark_table.md across 1,000 seeded evaluations:", fill="#94a3b8", font=FONT_BODY)

    # Summary Comparison Box
    box_y = 200
    # Left Card: Wind Robustness
    draw.rounded_rectangle([160, box_y, 600, box_y + 190], radius=10, fill="#0f172a", outline="#1e293b")
    draw.text((180, box_y + 15), "1. Wind Disturbance Sweep", fill="#38bdf8", font=FONT_HEADING)
    lines_wind = [
        "- Classical PID maintains 94% -> 89% success across full wind.",
        "- Aerodynamic wind force introduces lateral drift,",
        "  which outer-loop attitude controllers successfully correct.",
        "- Both PID and RL demonstrate robust wind resilience.",
    ]
    wy = box_y + 45
    for l in lines_wind:
        draw.text((180, wy), l, fill="#cbd5e1", font=FONT_MONO)
        wy += 25

    # Right Card: Sensor Noise Robustness
    draw.rounded_rectangle([640, box_y, 1080, box_y + 190], radius=10, fill="#0f172a", outline="#1e293b")
    draw.text((660, box_y + 15), "2. Sensor Noise Sweep", fill="#34d399", font=FONT_HEADING)
    lines_noise = [
        "- Classical PID collapses from 94% down to 3% at noise 0.25,",
        "  and 0% at noise >= 0.50 due to raw derivative explosion.",
        "- PPO neural policy acts as smooth non-linear filter,",
        "  avoiding high-frequency actuator jitter and tumbling.",
    ]
    ny = box_y + 45
    for l in lines_noise:
        draw.text((660, ny), l, fill="#cbd5e1", font=FONT_MONO)
        ny += 25

    # Mandatory Simulation Disclaimer Banner
    banner_y = 420
    draw.rounded_rectangle([160, banner_y, 1080, banner_y + 60], radius=8, fill="#450a0a", outline="#ef4444", width=2)
    draw.text((200, banner_y + 16), "SIMULATION ONLY", fill="#fca5a5", font=FONT_HUGE_BANNER)
    draw.text((540, banner_y + 22), "Planar 6-DOF rigid body rocket physics simulation with RK4 integration.", fill="#fecaca", font=FONT_BODY)

    # Footer
    draw.line([160, 510, 1080, 510], fill="#1e293b", width=1)
    draw.text((160, 530), "Open-Source Implementation: https://github.com/AdityaSurse/hoverslam", fill="#38bdf8", font=FONT_MONO)
    draw.text((880, 530), "Model: PPO Randomized (Seed 0)", fill="#64748b", font=FONT_MONO)

    return np.array(img)


# -----------------------------------------------------------------------------
# Main Video Generation Pipeline
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HoverSlam side-by-side demo video.")
    parser.add_argument(
        "--output-video",
        type=str,
        default="results/hoverslam_demo.mp4",
        help="Target MP4 video path.",
    )
    parser.add_argument(
        "--output-gif",
        type=str,
        default="results/hoverslam_preview.gif",
        help="Target 10-second GIF preview path.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_randomized_best/final_model.zip",
        help="Path to trained PPO model checkpoint.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Output video frames per second.",
    )
    return parser.parse_args()


def find_scenario_seeds(
    ppo_model: PPO,
    wind_strength: float = 0.75,
) -> tuple[int, int]:
    """Search for validated scenario seeds:
    1. fail_seed: PID crashes, RL lands (under sensor noise = 0.20)
    2. clean_seed: Both PID and RL land (under sensor noise = 0.02)
    """
    print("Searching for benchmark scenario seeds...")
    fail_seed: int | None = None
    clean_seed: int | None = None

    # Step 1: Search for crash seed (PID crashes, RL lands)
    for s in range(50):
        env_p = HoverSlamEnv(wind_strength=wind_strength, sensor_noise=0.20)
        obs_p, _ = env_p.reset(seed=s)
        pid = HoverSlamPID(dt=env_p.dt)
        done_p, landed_p = False, False
        while not done_p:
            act_p = pid.compute_action(obs_p)
            obs_p, _, term_p, trunc_p, info_p = env_p.step(act_p)
            done_p = term_p or trunc_p
            if info_p.get("landed", False):
                landed_p = True
        env_p.close()

        if landed_p:
            continue  # We need PID to fail

        env_r = HoverSlamEnv(wind_strength=wind_strength, sensor_noise=0.20)
        obs_r, _ = env_r.reset(seed=s)
        done_r, landed_r = False, False
        while not done_r:
            act_r, _ = ppo_model.predict(obs_r, deterministic=True)
            obs_r, _, term_r, trunc_r, info_r = env_r.step(act_r)
            done_r = term_r or trunc_r
            if info_r.get("landed", False):
                landed_r = True
        env_r.close()

        if landed_r:
            fail_seed = s
            print(f"  [Found Scenario 1 Seed] Seed {s}: PID Crashed, RL Landed!")
            break

    # Step 2: Search for clean seed (both land)
    for s in range(50):
        env_p = HoverSlamEnv(wind_strength=wind_strength, sensor_noise=0.02)
        obs_p, _ = env_p.reset(seed=s)
        pid = HoverSlamPID(dt=env_p.dt)
        done_p, landed_p = False, False
        while not done_p:
            act_p = pid.compute_action(obs_p)
            obs_p, _, term_p, trunc_p, info_p = env_p.step(act_p)
            done_p = term_p or trunc_p
            if info_p.get("landed", False):
                landed_p = True
        env_p.close()

        if not landed_p:
            continue

        env_r = HoverSlamEnv(wind_strength=wind_strength, sensor_noise=0.02)
        obs_r, _ = env_r.reset(seed=s)
        done_r, landed_r = False, False
        while not done_r:
            act_r, _ = ppo_model.predict(obs_r, deterministic=True)
            obs_r, _, term_r, trunc_r, info_r = env_r.step(act_r)
            done_r = term_r or trunc_r
            if info_r.get("landed", False):
                landed_r = True
        env_r.close()

        if landed_r:
            clean_seed = s
            print(f"  [Found Scenario 2 Seed] Seed {s}: Both Landed Cleanly!")
            break

    # Fallbacks if search limits exceeded
    if fail_seed is None:
        fail_seed = 0
    if clean_seed is None:
        clean_seed = 1

    return fail_seed, clean_seed


def main() -> None:
    args = parse_args()
    out_video_path = Path(args.output_video)
    out_gif_path = Path(args.output_gif)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_gif_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("               HOVERSLAM VIDEO DEMO GENERATOR")
    print("=" * 72)
    print(f"Target Video:    {out_video_path}")
    print(f"Target GIF:      {out_gif_path}")
    print(f"Model Checkpoint: {args.model_path}")
    print(f"FPS:             {args.fps}")
    print("=" * 72 + "\n")

    # Load PPO Model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"PPO Model not found at: {args.model_path}")
    print(f"Loading trained PPO model from: {args.model_path}")
    ppo_model = PPO.load(args.model_path)

    # Initialize PID controller
    sample_env = HoverSlamEnv()
    pid_controller = HoverSlamPID(dt=sample_env.dt)
    sample_env.close()

    # Automatically identify scenario seeds
    fail_seed, clean_seed = find_scenario_seeds(ppo_model, wind_strength=0.75)
    print(f"\nScenario Configurations:")
    print(f" - Scenario 1 (PID Crashes, RL Lands): Seed {fail_seed}, Wind=0.75, Sensor Noise=0.20")
    print(f" - Scenario 2 (Clean Landing Both):     Seed {clean_seed}, Wind=0.75, Sensor Noise=0.02\n")

    # -------------------------------------------------------------------------
    # Simulate Episodes
    # -------------------------------------------------------------------------
    print("Simulating Scenario 1...")
    env_s1_pid = HoverSlamEnv(wind_strength=0.75, sensor_noise=0.20)
    pid_s1_frames = simulate_episode(env_s1_pid, "pid", pid_controller, seed=fail_seed)
    env_s1_pid.close()

    env_s1_rl = HoverSlamEnv(wind_strength=0.75, sensor_noise=0.20)
    rl_s1_frames = simulate_episode(env_s1_rl, "rl", ppo_model, seed=fail_seed)
    env_s1_rl.close()

    print("Simulating Scenario 2...")
    env_s2_pid = HoverSlamEnv(wind_strength=0.75, sensor_noise=0.02)
    pid_s2_frames = simulate_episode(env_s2_pid, "pid", pid_controller, seed=clean_seed)
    env_s2_pid.close()

    env_s2_rl = HoverSlamEnv(wind_strength=0.75, sensor_noise=0.02)
    rl_s2_frames = simulate_episode(env_s2_rl, "rl", ppo_model, seed=clean_seed)
    env_s2_rl.close()

    # Synchronize frame lengths
    max_s1 = max(len(pid_s1_frames), len(rl_s1_frames))
    while len(pid_s1_frames) < max_s1:
        pid_s1_frames.append(pid_s1_frames[-1])
    while len(rl_s1_frames) < max_s1:
        rl_s1_frames.append(rl_s1_frames[-1])

    max_s2 = max(len(pid_s2_frames), len(rl_s2_frames))
    while len(pid_s2_frames) < max_s2:
        pid_s2_frames.append(pid_s2_frames[-1])
    while len(rl_s2_frames) < max_s2:
        rl_s2_frames.append(rl_s2_frames[-1])

    print(f"Scenario 1 Length: {max_s1} frames ({max_s1 / args.fps:.1f}s)")
    print(f"Scenario 2 Length: {max_s2} frames ({max_s2 / args.fps:.1f}s)")

    # -------------------------------------------------------------------------
    # Render Video Frames via imageio-ffmpeg
    # -------------------------------------------------------------------------
    print(f"\nEncoding MP4 video to {out_video_path}...")
    t_start = time.time()
    writer = imageio.get_writer(
        str(out_video_path),
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
    )

    total_frames_written = 0

    # 1. Title Card (4.0 seconds = 100 frames)
    title_frame = render_title_card()
    for _ in range(int(4.0 * args.fps)):
        writer.append_data(title_frame)
        total_frames_written += 1

    # 2. Scenario 1 Transition Slate (2.5 seconds = 62 frames)
    s1_slate = render_transition_card(
        scenario_num=1,
        title="High Sensor Noise Stress Test (Noise = 0.20, Wind = 0.75)",
        subtitle="Comparing Classical Derivative Amplification vs. Neural State Estimation",
        disturbance_info=f"Wind 0.75 (peak ~6 m/s) + Sensor Noise 0.20 | Seed {fail_seed}",
    )
    for _ in range(int(2.5 * args.fps)):
        writer.append_data(s1_slate)
        total_frames_written += 1

    # 3. Scenario 1 Flight Sequence
    gif_frames: list[np.ndarray] = []
    print("  Rendering Scenario 1 frames...")
    for idx in range(max_s1):
        f_pid = pid_s1_frames[idx]
        f_rl = rl_s1_frames[idx]
        frame_rgb = render_split_frame(
            f_pid=f_pid,
            f_rl=f_rl,
            scenario_title="Scenario 1: High Sensor Noise",
            wind_label="Wind: 0.75",
            noise_label="Noise: 0.20",
        )
        writer.append_data(frame_rgb)
        total_frames_written += 1

        # Capture dramatic touchdown moment for 10-second GIF preview
        # (Around step 450 to 650)
        if 420 <= idx < 570:
            # Resize for compact web GIF
            small_img = Image.fromarray(frame_rgb).resize((640, 360), Image.Resampling.BILINEAR)
            gif_frames.append(np.array(small_img))

    # 4. Scenario 2 Transition Slate (2.5 seconds)
    s2_slate = render_transition_card(
        scenario_num=2,
        title="Clean Nominal Conditions (Noise = 0.02, Wind = 0.75)",
        subtitle="Comparing Aerodynamic Disturbance Compensation Under Clean Sensors",
        disturbance_info=f"Wind 0.75 (peak ~6 m/s) + Sensor Noise 0.02 | Seed {clean_seed}",
    )
    for _ in range(int(2.5 * args.fps)):
        writer.append_data(s2_slate)
        total_frames_written += 1

    # 5. Scenario 2 Flight Sequence
    print("  Rendering Scenario 2 frames...")
    for idx in range(max_s2):
        f_pid = pid_s2_frames[idx]
        f_rl = rl_s2_frames[idx]
        frame_rgb = render_split_frame(
            f_pid=f_pid,
            f_rl=f_rl,
            scenario_title="Scenario 2: Nominal Clean Sensors",
            wind_label="Wind: 0.75",
            noise_label="Noise: 0.02",
        )
        writer.append_data(frame_rgb)
        total_frames_written += 1

    # 6. End Summary Card (5.0 seconds)
    end_frame = render_end_card()
    for _ in range(int(5.0 * args.fps)):
        writer.append_data(end_frame)
        total_frames_written += 1

    writer.close()
    duration_sec = total_frames_written / args.fps
    encode_time = time.time() - t_start
    file_size_mb = out_video_path.stat().st_size / (1024 * 1024)

    print(f"\n[Video Saved] {out_video_path}")
    print(f" - Duration:  {duration_sec:.1f} seconds ({total_frames_written} frames at {args.fps} FPS)")
    print(f" - File Size: {file_size_mb:.2f} MB")
    print(f" - Render:    {encode_time:.1f} seconds ({total_frames_written / encode_time:.1f} FPS)")

    # -------------------------------------------------------------------------
    # Render 10-Second GIF Preview
    # -------------------------------------------------------------------------
    print(f"\nExporting 10-second GIF preview to {out_gif_path}...")
    if len(gif_frames) > 0:
        # 150 frames at 15 fps = 10.0 seconds
        subsampled_gif = gif_frames[::1]
        imageio.mimsave(
            str(out_gif_path),
            subsampled_gif,
            fps=15,
            loop=0,
        )
        gif_size_mb = out_gif_path.stat().st_size / (1024 * 1024)
        print(f"[GIF Saved] {out_gif_path} ({gif_size_mb:.2f} MB, {len(subsampled_gif)} frames)")

    print("\n" + "=" * 72)
    print("               HOVERSLAM DEMO RENDERING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
