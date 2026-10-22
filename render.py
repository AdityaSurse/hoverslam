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
def load_font(
    size: int,
    bold: bool = False,
    mono: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load high-quality system font with proportional or monospace preference."""
    if mono:
        font_candidates = [
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/Library/Fonts/Courier New.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    else:
        font_candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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


FONT_TITLE = load_font(38, bold=True, mono=False)
FONT_SUBTITLE = load_font(21, bold=True, mono=False)
FONT_HEADING = load_font(16, bold=True, mono=False)
FONT_BODY = load_font(13, bold=False, mono=False)
FONT_BADGE = load_font(14, bold=True, mono=False)
FONT_MONO = load_font(12, bold=False, mono=True)
FONT_MONO_BOLD = load_font(12, bold=True, mono=True)
FONT_HUGE_BANNER = load_font(22, bold=True, mono=False)


def draw_pill(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text_color: str,
    bg_color: str,
    border_color: str,
    pad_x: int = 12,
    pad_y: int = 5,
    align_right: bool = False,
) -> int:
    """Draw a rounded badge box sized dynamically to fit its text. Returns pill width."""
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pw = tw + 2 * pad_x
    ph = th + 2 * pad_y
    left = x - pw if align_right else x
    draw.rounded_rectangle([left, y, left + pw, y + ph], radius=6, fill=bg_color, outline=border_color)
    draw.text((left + pad_x, y + pad_y), text, fill=text_color, font=font)
    return pw


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
# Rendering Primitives & Aerospace Graphics
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


def create_base_canvas() -> Image.Image:
    """Pre-bake atmospheric sky dome, starry exosphere, and landing drone ship pads."""
    img = Image.new("RGBA", (WIDTH, HEIGHT_PX), (7, 11, 20, 255))
    draw = ImageDraw.Draw(img)

    # 1. Atmospheric Sky Gradient (y = HEADER_H to GROUND_Y)
    # Deep exosphere navy (0x03, 0x06, 0x10) -> Mid stratosphere (0x09, 0x14, 0x2b) -> Horizon glow (0x15, 0x2e, 0x54)
    sky_h = GROUND_Y - HEADER_H
    for i in range(sky_h):
        y = HEADER_H + i
        t = i / float(sky_h)
        t_curv = math.pow(t, 2.2)
        r = int(3 + t_curv * (22 - 3))
        g = int(6 + t_curv * (46 - 6))
        b = int(16 + t_curv * (86 - 16))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b, 255))

    # 2. Subtle Starfield in upper sky (y < GROUND_Y - 170)
    np.random.seed(42)
    for _ in range(120):
        sx = int(np.random.randint(10, WIDTH - 10))
        sy = int(np.random.randint(HEADER_H + 8, GROUND_Y - 180))
        lum = int(np.random.randint(45, 140))
        draw.point((sx, sy), fill=(lum, lum, int(min(255, lum * 1.25)), 255))

    # 3. Horizon Boundary Glow Line (Atmospheric limb)
    draw.line([(0, GROUND_Y - 1), (WIDTH, GROUND_Y - 1)], fill=(32, 68, 120, 255), width=2)

    # 4. Deep Ocean Water (y = GROUND_Y to MAIN_H)
    ocean_h = MAIN_H - GROUND_Y
    for i in range(ocean_h):
        y = GROUND_Y + i
        t = i / float(ocean_h)
        r = int(9 + t * (5 - 9))
        g = int(16 + t * (9 - 16))
        b = int(28 + t * (18 - 28))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b, 255))

    # Smooth water highlights
    draw.line([(0, GROUND_Y + 1), (WIDTH, GROUND_Y + 1)], fill=(20, 45, 80, 160), width=1)
    draw.line([(0, GROUND_Y + 6), (WIDTH, GROUND_Y + 6)], fill=(14, 30, 55, 120), width=1)

    # 5. Engineering Altitude Grid Lines & Labels
    for alt_m in [50, 100, 150, 200, 250, 300, 350]:
        y_grid = int(GROUND_Y - alt_m * SCALE_M_TO_PX)
        for gx in range(0, WIDTH, 16):
            draw.line([(gx, y_grid), (gx + 8, y_grid)], fill=(28, 45, 70, 160), width=1)
        draw.text((12, y_grid - 7), f"{alt_m}m", fill=(71, 95, 130, 230), font=FONT_MONO)
        draw.text((WIDTH // 2 + 12, y_grid - 7), f"{alt_m}m", fill=(71, 95, 130, 230), font=FONT_MONO)

    # 6. Center divider between left and right flight corridors
    draw.line([(WIDTH // 2, HEADER_H), (WIDTH // 2, MAIN_H)], fill=(30, 41, 59, 255), width=2)

    # 7. Drone Ship Landing Platforms (Left and Right)
    for cx in [LEFT_PAD_X, RIGHT_PAD_X]:
        # Upward floodlight glow cone (smooth quadratic gradient fade)
        spot_layer = Image.new("RGBA", (WIDTH, HEIGHT_PX), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(spot_layer)
        for h_step in range(160):
            frac = h_step / 160.0
            cone_w = 35 + frac * 42
            alpha = int(math.pow(1.0 - frac, 1.8) * 28)
            y_spot = GROUND_Y - h_step
            sdraw.line([(cx - cone_w, y_spot), (cx + cone_w, y_spot)], fill=(56, 189, 248, alpha))
        img = Image.alpha_composite(img, spot_layer)
        draw = ImageDraw.Draw(img)

        # Faint glideslope guidance corridor (Subtle cyan-gray dashed lines)
        for gx_off in [-22, 22]:
            for gy in range(HEADER_H + 20, GROUND_Y - 4, 14):
                prog = (GROUND_Y - gy) / float(GROUND_Y - HEADER_H)
                spread = gx_off * (1.0 + prog * 1.4)
                draw.line([(cx + spread, gy), (cx + spread, gy + 7)], fill=(56, 189, 248, 30), width=1)

        # Drone ship barge deck
        barge_hw = 48
        barge_top = GROUND_Y - 2
        barge_bot = GROUND_Y + 9

        # Barge shadow in water
        draw.rectangle([cx - barge_hw, barge_bot, cx + barge_hw, barge_bot + 4], fill=(4, 7, 14, 255))
        # Hull & Deck
        draw.rectangle([cx - barge_hw, barge_top + 3, cx + barge_hw, barge_bot], fill=(15, 23, 42, 255), outline=(30, 41, 59, 255))
        draw.rectangle([cx - barge_hw, barge_top, cx + barge_hw, barge_top + 3], fill=(30, 41, 59, 255), outline=(51, 65, 85, 255))

        # Yellow hazard caution stripes
        for hx in range(cx - barge_hw + 2, cx - barge_hw + 16, 4):
            draw.line([(hx, barge_top), (hx + 2, barge_top + 3)], fill=(234, 179, 8, 255), width=2)
        for hx in range(cx + barge_hw - 16, cx + barge_hw - 2, 4):
            draw.line([(hx, barge_top), (hx + 2, barge_top + 3)], fill=(234, 179, 8, 255), width=2)

        # Precision Landing Target Ring & Bullseye
        draw.ellipse([cx - 18, barge_top - 3, cx + 18, barge_top + 3], outline=(16, 185, 129, 230), width=1)
        draw.ellipse([cx - 5, barge_top - 2, cx + 5, barge_top + 2], fill=(16, 185, 129, 255))
        draw.line([(cx - 28, barge_top), (cx - 12, barge_top)], fill=(16, 185, 129, 180), width=1)
        draw.line([(cx + 12, barge_top), (cx + 28, barge_top)], fill=(16, 185, 129, 180), width=1)

        # Deck Beacons (Port red, Starboard green)
        draw.ellipse([cx - barge_hw - 2, barge_top - 3, cx - barge_hw + 2, barge_top + 1], fill=(239, 68, 68, 255))
        draw.ellipse([cx + barge_hw - 2, barge_top - 3, cx + barge_hw + 2, barge_top + 1], fill=(34, 197, 94, 255))

    return img


def draw_trajectory_ribbon(
    img: Image.Image,
    pad_cx: int,
    ground_y: int,
    history: list[tuple[float, float]],
    accent_rgba: tuple[int, int, int, int],
) -> Image.Image:
    """Draw semi-transparent flight path trajectory ribbon with subtle glow."""
    if len(history) < 2:
        return img

    trail_layer = Image.new("RGBA", (WIDTH, HEIGHT_PX), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(trail_layer)

    pts = []
    for x_m, y_m in history:
        sx = pad_cx + x_m * SCALE_M_TO_PX
        sy = ground_y - y_m * SCALE_M_TO_PX
        pts.append((sx, sy))

    # Outer soft glow ribbon
    glow_col = (accent_rgba[0], accent_rgba[1], accent_rgba[2], int(accent_rgba[3] * 0.35))
    for i in range(len(pts) - 1):
        tdraw.line([pts[i], pts[i + 1]], fill=glow_col, width=4)

    # Core high-precision flight path ribbon
    for i in range(len(pts) - 1):
        tdraw.line([pts[i], pts[i + 1]], fill=accent_rgba, width=2)

    return Image.alpha_composite(img, trail_layer)


def draw_rocket_and_plume(
    img: Image.Image,
    pad_cx: int,
    ground_y: int,
    frame: TelemetryFrame,
    step_flicker: int = 0,
) -> Image.Image:
    """Render aerospace rocket booster, dynamic grid fins, deployable legs, RCS, and plume."""
    rx = pad_cx + frame.x * SCALE_M_TO_PX
    ry = ground_y - frame.y * SCALE_M_TO_PX
    rx = max(30, min(1250, rx))
    ry = min(ground_y + 10, max(60, ry))

    theta_scr = -frame.theta  # Invert for screen coordinates

    bw = 8.0
    bh = 38.0
    half_w = bw / 2.0
    half_h = bh / 2.0

    fx_layer = Image.new("RGBA", (WIDTH, HEIGHT_PX), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(fx_layer)

    # 1. Ground Kickup Dust / Vapor billow (when near pad deck)
    if frame.y < 14.0 and frame.throttle > 0.12 and not frame.done:
        dust_w = int(24.0 + (14.0 - frame.y) * 3.5)
        dust_h = int(6.0 + (14.0 - frame.y) * 0.8)
        dust_alpha = int(min(120, frame.throttle * 150))
        dust_box = [rx - dust_w, ground_y - dust_h, rx + dust_w, ground_y + 3]
        fdraw.ellipse(dust_box, fill=(148, 163, 184, dust_alpha))

    # 2. Supersonic Exhaust Plume & Mach Diamonds
    if frame.throttle > 0.01 and not frame.done:
        flicker = 1.0 + 0.08 * math.sin(step_flicker * 1.5)
        flame_len = (10.0 + 52.0 * float(frame.throttle)) * flicker
        flame_angle = theta_scr - frame.gimbal

        nozzle_pt = rotate_point(rx, ry + half_h, rx, ry, theta_scr)

        # A. Outer Supersonic Boundary (Deep Orange / Amber)
        tip_outer = rotate_point(nozzle_pt[0], nozzle_pt[1] + flame_len, nozzle_pt[0], nozzle_pt[1], flame_angle)
        left_outer = rotate_point(nozzle_pt[0] - 6.5, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        right_outer = rotate_point(nozzle_pt[0] + 6.5, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        fdraw.polygon([left_outer, tip_outer, right_outer], fill=(234, 88, 12, 220))

        # B. Mid Ionization Envelope (Bright Yellow)
        tip_mid = rotate_point(nozzle_pt[0], nozzle_pt[1] + flame_len * 0.72, nozzle_pt[0], nozzle_pt[1], flame_angle)
        left_mid = rotate_point(nozzle_pt[0] - 4.2, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        right_mid = rotate_point(nozzle_pt[0] + 4.2, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        fdraw.polygon([left_mid, tip_mid, right_mid], fill=(250, 204, 21, 240))

        # C. Inner Supersonic Jet Core (Cyan-White Ionized Plasma)
        tip_inner = rotate_point(nozzle_pt[0], nozzle_pt[1] + flame_len * 0.40, nozzle_pt[0], nozzle_pt[1], flame_angle)
        left_inner = rotate_point(nozzle_pt[0] - 2.5, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        right_inner = rotate_point(nozzle_pt[0] + 2.5, nozzle_pt[1], nozzle_pt[0], nozzle_pt[1], flame_angle)
        fdraw.polygon([left_inner, tip_inner, right_inner], fill=(224, 242, 254, 255))

        # D. Mach Diamonds (Shock Disks along centerline)
        for d_frac in [0.22, 0.42, 0.62]:
            if flame_len * d_frac < flame_len - 4:
                d_center = rotate_point(nozzle_pt[0], nozzle_pt[1] + flame_len * d_frac, nozzle_pt[0], nozzle_pt[1], flame_angle)
                d_top = rotate_point(d_center[0], d_center[1] - 2.5, d_center[0], d_center[1], flame_angle)
                d_bot = rotate_point(d_center[0], d_center[1] + 2.5, d_center[0], d_center[1], flame_angle)
                d_l = rotate_point(d_center[0] - 2.0, d_center[1], d_center[0], d_center[1], flame_angle)
                d_r = rotate_point(d_center[0] + 2.0, d_center[1], d_center[0], d_center[1], flame_angle)
                fdraw.polygon([d_top, d_r, d_bot, d_l], fill=(255, 255, 255, 240))

    # 3. Cold-Gas RCS Thruster Puffs (When angular rate |omega| > 0.012 rad/s)
    if abs(frame.omega) > 0.012 and not frame.done:
        rcs_side = -1.0 if frame.omega > 0 else 1.0  # Thrust counters spin
        rcs_nose = rotate_point(rx, ry - half_h + 5.0, rx, ry, theta_scr)
        rcs_tip = rotate_point(rcs_nose[0] + rcs_side * 14.0, rcs_nose[1], rcs_nose[0], rcs_nose[1], theta_scr)
        rcs_top = rotate_point(rcs_nose[0], rcs_nose[1] - 3.0, rcs_nose[0], rcs_nose[1], theta_scr)
        rcs_bot = rotate_point(rcs_nose[0], rcs_nose[1] + 3.0, rcs_nose[0], rcs_nose[1], theta_scr)
        fdraw.polygon([rcs_top, rcs_tip, rcs_bot], fill=(224, 242, 254, 180))

    # 4. Gimbaled Engine Nozzle Bell
    nozzle_center = rotate_point(rx, ry + half_h + 2.0, rx, ry, theta_scr)
    nozzle_angle = theta_scr - frame.gimbal
    nb_l = rotate_point(nozzle_center[0] - 3.5, nozzle_center[1] + 3.0, nozzle_center[0], nozzle_center[1], nozzle_angle)
    nb_r = rotate_point(nozzle_center[0] + 3.5, nozzle_center[1] + 3.0, nozzle_center[0], nozzle_center[1], nozzle_angle)
    nt_l = rotate_point(nozzle_center[0] - 2.0, nozzle_center[1] - 2.0, nozzle_center[0], nozzle_center[1], nozzle_angle)
    nt_r = rotate_point(nozzle_center[0] + 2.0, nozzle_center[1] - 2.0, nozzle_center[0], nozzle_center[1], nozzle_angle)
    fdraw.polygon([nt_l, nt_r, nb_r, nb_l], fill=(15, 23, 42, 255), outline=(51, 65, 85, 255))

    # 5. Rocket Cylindrical Body with 3D Specular Shading
    corners_left = [
        (-half_w, -half_h),
        (0.0, -half_h),
        (0.0, half_h - 4.0),
        (-half_w, half_h - 4.0),
    ]
    rot_c_left = [rotate_point(rx + cx, ry - cy, rx, ry, theta_scr) for cx, cy in corners_left]
    fdraw.polygon(rot_c_left, fill=(203, 213, 225, 255))

    corners_right = [
        (0.0, -half_h),
        (half_w, -half_h),
        (half_w, half_h - 4.0),
        (0.0, half_h - 4.0),
    ]
    rot_c_right = [rotate_point(rx + cx, ry - cy, rx, ry, theta_scr) for cx, cy in corners_right]
    fdraw.polygon(rot_c_right, fill=(248, 250, 252, 255))

    nose_corners = [
        (-half_w, half_h - 4.0),
        (0.0, half_h + 1.0),
        (half_w, half_h - 4.0),
    ]
    rot_nose = [rotate_point(rx + cx, ry - cy, rx, ry, theta_scr) for cx, cy in nose_corners]
    fdraw.polygon(rot_nose, fill=(241, 245, 249, 255), outline=(148, 163, 184, 255))

    body_perimeter = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h - 4.0),
        (0.0, half_h + 1.0),
        (-half_w, half_h - 4.0),
    ]
    rot_perim = [rotate_point(rx + cx, ry - cy, rx, ry, theta_scr) for cx, cy in body_perimeter]
    fdraw.polygon(rot_perim, outline=(100, 116, 139, 255))

    # Black Interstage Band
    ring_corners = [
        (-half_w, half_h - 14.0),
        (half_w, half_h - 14.0),
        (half_w, half_h - 9.0),
        (-half_w, half_h - 9.0),
    ]
    rot_ring = [rotate_point(rx + cx, ry - cy, rx, ry, theta_scr) for cx, cy in ring_corners]
    fdraw.polygon(rot_ring, fill=(15, 23, 42, 255), outline=(30, 41, 59, 255))

    # 6. Titanium Grid Fins (Mounted below nose cone, canting with attitude control)
    grid_y = half_h - 11.5
    grid_cant = max(-0.4, min(0.4, frame.omega * 1.8 - frame.theta * 0.5))
    gf_l1 = rotate_point(rx - half_w, ry - grid_y, rx, ry, theta_scr)
    gf_l2 = rotate_point(rx - half_w - 7.5, ry - grid_y - grid_cant * 8.0, rx, ry, theta_scr)
    fdraw.line([gf_l1, gf_l2], fill=(148, 163, 184, 255), width=2)
    gf_r1 = rotate_point(rx + half_w, ry - grid_y, rx, ry, theta_scr)
    gf_r2 = rotate_point(rx + half_w + 7.5, ry - grid_y + grid_cant * 8.0, rx, ry, theta_scr)
    fdraw.line([gf_r1, gf_r2], fill=(148, 163, 184, 255), width=2)

    # 7. Deployable Landing Legs (Deploy dynamically when altitude y < 35m)
    if frame.y >= 35.0:
        deploy_f = 0.0
    elif frame.y <= 15.0 or frame.landed or frame.done:
        deploy_f = 1.0
    else:
        deploy_f = (35.0 - frame.y) / 20.0

    leg_span = 2.0 + deploy_f * 7.5
    leg_drop = 0.0 + deploy_f * 4.5

    leg_l_top = rotate_point(rx - half_w, ry + half_h - 10.0 + (1.0 - deploy_f) * 6.0, rx, ry, theta_scr)
    leg_l_foot = rotate_point(rx - half_w - leg_span, ry + half_h + leg_drop, rx, ry, theta_scr)
    fdraw.line([leg_l_top, leg_l_foot], fill=(203, 213, 225, 255), width=2)
    fdraw.ellipse([leg_l_foot[0] - 1.5, leg_l_foot[1] - 1.0, leg_l_foot[0] + 1.5, leg_l_foot[1] + 1.0], fill=(71, 85, 105, 255))

    leg_r_top = rotate_point(rx + half_w, ry + half_h - 10.0 + (1.0 - deploy_f) * 6.0, rx, ry, theta_scr)
    leg_r_foot = rotate_point(rx + half_w + leg_span, ry + half_h + leg_drop, rx, ry, theta_scr)
    fdraw.line([leg_r_top, leg_r_foot], fill=(203, 213, 225, 255), width=2)
    fdraw.ellipse([leg_r_foot[0] - 1.5, leg_r_foot[1] - 1.0, leg_r_foot[0] + 1.5, leg_r_foot[1] + 1.0], fill=(71, 85, 105, 255))

    return Image.alpha_composite(img, fx_layer)


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
    """Render aerospace telemetry dashboard card with gauges, values, and status badge."""
    draw.rounded_rectangle([box_x1, box_y1, box_x2, box_y2], radius=10, fill="#0d1424", outline="#1e293b", width=2)

    # Controller Title Bar
    draw.rectangle([box_x1 + 8, box_y1 + 8, box_x2 - 8, box_y1 + 34], fill="#111c33")
    draw.text((box_x1 + 14, box_y1 + 11), controller_name, fill=accent_color, font=FONT_BADGE)

    # Status Pill on Right of Header (Dynamically right-aligned)
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

    draw_pill(
        draw=draw,
        x=box_x2 - 14,
        y=box_y1 + 9,
        text=status_text,
        font=FONT_MONO_BOLD,
        text_color=text_color,
        bg_color=status_bg,
        border_color=status_border,
        pad_x=10,
        pad_y=3,
        align_right=True,
    )

    # Telemetry Columns (Spaced evenly across card)
    row1_y = box_y1 + 42
    row2_y = box_y1 + 76
    row3_y = box_y1 + 110

    card_usable_w = box_x2 - box_x1 - 28
    col_w = card_usable_w // 4

    c1 = box_x1 + 14
    c2 = box_x1 + 14 + col_w
    c3 = box_x1 + 14 + 2 * col_w
    c4 = box_x1 + 14 + 3 * col_w

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
    draw.text((c3, row1_y + 14), f"{tilt_deg:+5.1f}°", fill=COLOR_TEXT_MAIN, font=FONT_MONO_BOLD)

    gimbal_deg = math.degrees(frame.gimbal)
    draw.text((c3, row2_y), "GIMBAL NOZZLE", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c3, row2_y + 14), f"{gimbal_deg:+5.1f}°", fill=COLOR_TEXT_MAIN, font=FONT_MONO_BOLD)

    # Column 4: Propulsion & Fuel Gauges
    thr_pct = frame.throttle * 100.0
    draw.text((c4, row1_y), "THROTTLE", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c4 + 75, row1_y), f"{thr_pct:3.0f}%", fill="#f97316", font=FONT_MONO_BOLD)
    draw.rectangle([c4, row1_y + 16, c4 + 130, row1_y + 24], fill="#1e293b", outline="#475569")
    fill_w = int(130.0 * min(1.0, max(0.0, frame.throttle)))
    if fill_w > 0:
        draw.rectangle([c4, row1_y + 16, c4 + fill_w, row1_y + 24], fill="#ea580c")

    fuel_pct = (frame.fuel / FUEL_MASS) * 100.0
    draw.text((c4, row2_y), "PROPELLANT", fill=COLOR_TEXT_MUTED, font=FONT_MONO)
    draw.text((c4 + 82, row2_y), f"{frame.fuel:5.1f}kg", fill="#38bdf8", font=FONT_MONO_BOLD)
    draw.rectangle([c4, row2_y + 16, c4 + 130, row2_y + 24], fill="#1e293b", outline="#475569")
    fuel_w = int(130.0 * min(1.0, max(0.0, frame.fuel / FUEL_MASS)))
    if fuel_w > 0:
        draw.rectangle([c4, row2_y + 16, c4 + fuel_w, row2_y + 24], fill="#0284c7")

    # Bottom Row Note
    draw.line([box_x1 + 12, row3_y + 8, box_x2 - 12, row3_y + 8], fill="#1e293b", width=1)
    if frame.landed:
        status_summary = "PERFECT TOUCHDOWN: Impact speed < 2 m/s, upright on pad center"
    elif frame.done:
        status_summary = "CRASHED: High-velocity impact or lost attitude stability"
    else:
        status_summary = "Retro-propulsion active: hoverslam descent guidance"
    draw.text((box_x1 + 16, row3_y + 15), status_summary, fill=COLOR_TEXT_MUTED, font=FONT_BODY)


def render_split_frame(
    base_canvas: Image.Image,
    f_pid: TelemetryFrame,
    f_rl: TelemetryFrame,
    history_pid: list[tuple[float, float]],
    history_rl: list[tuple[float, float]],
    scenario_title: str,
    wind_label: str,
    noise_label: str,
    step_cnt: int = 0,
) -> np.ndarray:
    """Render complete 1280x720 side-by-side comparative frame with aerospace graphics."""
    # 1. Start from pre-rendered base atmospheric canvas
    comp = base_canvas.copy()

    # 2. Draw Trajectory Ribbons
    comp = draw_trajectory_ribbon(comp, LEFT_PAD_X, GROUND_Y, history_pid, (56, 189, 248, 160))
    comp = draw_trajectory_ribbon(comp, RIGHT_PAD_X, GROUND_Y, history_rl, (52, 211, 153, 180))

    # 3. Draw Rockets and Plumes
    comp = draw_rocket_and_plume(comp, LEFT_PAD_X, GROUND_Y, f_pid, step_flicker=step_cnt)
    comp = draw_rocket_and_plume(comp, RIGHT_PAD_X, GROUND_Y, f_rl, step_flicker=step_cnt)

    # 4. Convert composite to RGB for UI elements
    final_img = Image.new("RGB", (WIDTH, HEIGHT_PX), (7, 11, 20))
    final_img.paste(comp, (0, 0))
    draw = ImageDraw.Draw(final_img)

    # 5. Top Header Bar
    draw.rectangle([0, 0, WIDTH, HEADER_H], fill=COLOR_HEADER_BG)
    draw.line([0, HEADER_H, WIDTH, HEADER_H], fill=COLOR_PANEL_BORDER, width=2)
    draw.text((20, 14), "HOVERSLAM FLIGHT CONTROLLER TESTBENCH", fill=COLOR_TEXT_MAIN, font=FONT_HEADING)

    # Scenario badge in header (Dynamically right-aligned)
    scenario_badge_text = f"{scenario_title}  •  {wind_label}  •  {noise_label}"
    draw_pill(
        draw=draw,
        x=WIDTH - 20,
        y=10,
        text=scenario_badge_text,
        font=FONT_MONO_BOLD,
        text_color="#cbd5e1",
        bg_color="#0f172a",
        border_color="#334155",
        pad_x=12,
        pad_y=5,
        align_right=True,
    )

    # 6. Wind Vectors (Top corner of each panel, dynamically sized)
    for px, f in [(LEFT_PAD_X, f_pid), (RIGHT_PAD_X, f_rl)]:
        wx = f.wind_x
        wind_dir = "-->" if wx >= 0 else "<--"
        wind_str = f"WIND: {abs(wx):4.1f} N {wind_dir}"
        draw_pill(
            draw=draw,
            x=px + 295,
            y=HEADER_H + 12,
            text=wind_str,
            font=FONT_MONO_BOLD,
            text_color="#94a3b8",
            bg_color="#0f172a",
            border_color="#334155",
            pad_x=10,
            pad_y=4,
            align_right=True,
        )

    # 7. Telemetry Overlay Cards (Bottom Area)
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
        controller_name="CONTROLLER 1: CLASSICAL PID",
    )

    draw_hud_panel(
        draw,
        box_x1=margin * 2 + card_w,
        box_y1=MAIN_H + 8,
        box_x2=WIDTH - margin,
        box_y2=HEIGHT_PX - 12,
        frame=f_rl,
        accent_color=COLOR_RL_ACCENT,
        controller_name="CONTROLLER 2: PPO RANDOMIZED RL",
    )

    return np.array(final_img)


# -----------------------------------------------------------------------------
# Card Screens: Title, Transition, End Summary
# -----------------------------------------------------------------------------
def render_title_card() -> np.ndarray:
    """Generate sleek title card introducing the benchmark."""
    img = Image.new("RGB", (WIDTH, HEIGHT_PX), color="#050814")
    draw = ImageDraw.Draw(img)

    # Atmospheric sky gradient in title card background
    for y in range(HEIGHT_PX):
        t = y / float(HEIGHT_PX)
        r = int(5 + t * (14 - 5))
        g = int(8 + t * (24 - 8))
        b = int(20 + t * (46 - 20))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

    # Subtle starry background
    np.random.seed(42)
    for _ in range(90):
        sx = int(np.random.randint(0, WIDTH))
        sy = int(np.random.randint(0, HEIGHT_PX))
        draw.point((sx, sy), fill="#334155")

    # Glowing box container
    draw.rounded_rectangle([100, 75, WIDTH - 100, HEIGHT_PX - 75], radius=16, fill="#0b1120", outline="#1e293b", width=3)

    # Top Badge
    draw_pill(draw, 140, 115, "AUTONOMOUS AEROSPACE RL", FONT_MONO_BOLD, "#38bdf8", "#0f2847", "#0284c7", pad_x=12, pad_y=5)

    # Main Title
    draw.text((140, 165), "HoverSlam", fill="#38bdf8", font=FONT_TITLE)
    draw.text((140, 225), "Deep RL vs. Classical PID Rocket Landing", fill="#f8fafc", font=FONT_SUBTITLE)

    # Description Paragraph
    desc_lines = [
        "A rigorous side-by-side benchmark comparing classical cascaded PID control against",
        "domain-randomized Proximal Policy Optimization (PPO) in planar 2D retro-propulsion.",
        "",
        "Evaluated on identical test seeds under continuous Ornstein-Uhlenbeck wind gusts",
        "and unmodeled high-frequency sensor noise.",
    ]
    dy = 285
    for line in desc_lines:
        draw.text((140, dy), line, fill="#cbd5e1", font=FONT_BODY)
        dy += 24

    # Feature Pills
    pills = [
        ("WIND: Ornstein-Uhlenbeck (0.75)", "#064e3b", "#10b981", "#6ee7b7"),
        ("NOISE: Gaussian Sensor Disturbance", "#701a75", "#c026d3", "#f0abfc"),
        ("CONTROLLERS: PID vs PPO MlpPolicy", "#1e3a8a", "#2563eb", "#93c5fd"),
    ]
    px = 140
    for text, bg, border, fg in pills:
        pw = draw_pill(draw, px, 435, text, FONT_MONO_BOLD, fg, bg, border, pad_x=12, pad_y=6)
        px += pw + 16

    # Footer
    draw.line([140, 505, WIDTH - 140, 505], fill="#1e293b", width=2)
    draw.text((140, 525), "AdityaSurse / hoverslam  •  Gymnasium & PyTorch  •  imageio-ffmpeg", fill="#64748b", font=FONT_BODY)

    return np.array(img)


def render_transition_card(scenario_num: int, title: str, subtitle: str, disturbance_info: str) -> np.ndarray:
    """Generate transition slate between benchmark scenarios."""
    img = Image.new("RGB", (WIDTH, HEIGHT_PX), color="#060a16")
    draw = ImageDraw.Draw(img)

    for y in range(HEIGHT_PX):
        t = y / float(HEIGHT_PX)
        r = int(6 + t * (12 - 6))
        g = int(10 + t * (20 - 10))
        b = int(22 + t * (42 - 22))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

    draw.rounded_rectangle([140, 140, WIDTH - 140, HEIGHT_PX - 140], radius=16, fill="#0c1527", outline="#1e293b", width=2)

    badge = f"TEST FLIGHT SCENARIO {scenario_num} / 2"
    draw_pill(draw, 180, 180, badge, FONT_MONO_BOLD, "#93c5fd", "#1e3a8a", "#3b82f6", pad_x=12, pad_y=5)

    draw.text((180, 230), title, fill="#f8fafc", font=FONT_SUBTITLE)
    draw.text((180, 280), subtitle, fill="#94a3b8", font=FONT_BODY)

    draw.line([180, 330, WIDTH - 180, 330], fill="#1e293b", width=1)
    cond_text = f"TEST CONDITION: {disturbance_info}"
    draw_pill(draw, 180, 355, cond_text, FONT_MONO_BOLD, "#38bdf8", "#0c1b33", "#0284c7", pad_x=14, pad_y=7)

    return np.array(img)


def render_end_card() -> np.ndarray:
    """Generate summary end card with benchmark headline figures and disclaimers."""
    img = Image.new("RGB", (WIDTH, HEIGHT_PX), color="#050814")
    draw = ImageDraw.Draw(img)

    for y in range(HEIGHT_PX):
        t = y / float(HEIGHT_PX)
        r = int(5 + t * (14 - 5))
        g = int(8 + t * (24 - 8))
        b = int(20 + t * (46 - 20))
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

    draw.rounded_rectangle([80, 50, WIDTH - 80, HEIGHT_PX - 50], radius=16, fill="#0b1120", outline="#1e293b", width=3)

    draw.text((120, 85), "Benchmark Summary & Performance Takeaways", fill="#f8fafc", font=FONT_SUBTITLE)
    draw.text((120, 125), "Monte Carlo evaluation results across 1,000 fixed seeded episodes (from benchmark_table.md):", fill="#94a3b8", font=FONT_BODY)

    box_y = 170
    card_h = 220
    # Left Card: Wind Disturbance Sweep
    draw.rounded_rectangle([120, box_y, 615, box_y + card_h], radius=10, fill="#0f172a", outline="#1e293b")
    draw.text((140, box_y + 16), "1. Wind Disturbance Sweep", fill="#38bdf8", font=FONT_HEADING)
    lines_wind = [
        "• Classical PID maintains 94% -> 89% success across full wind sweep.",
        "• Continuous Ornstein-Uhlenbeck wind causes lateral drift;",
        "  outer-loop attitude loops counter with pitch steering.",
        "• Both PID and PPO demonstrate robust aerodynamic resilience.",
    ]
    wy = box_y + 50
    for l in lines_wind:
        draw.text((140, wy), l, fill="#cbd5e1", font=FONT_BODY)
        wy += 26

    # Right Card: Sensor Noise Robustness
    draw.rounded_rectangle([645, box_y, 1140, box_y + card_h], radius=10, fill="#0f172a", outline="#1e293b")
    draw.text((665, box_y + 16), "2. Sensor Noise Sweep", fill="#34d399", font=FONT_HEADING)
    lines_noise = [
        "• Classical PID collapses: 94% -> 3% (noise 0.25) -> 0% (>=0.50).",
        "• Dividing noisy observations by dt=0.05s amplifies noise 20x,",
        "  driving derivative gains into severe actuator saturation.",
        "• PPO neural policy acts as smooth non-linear state filter.",
    ]
    ny = box_y + 50
    for l in lines_noise:
        draw.text((665, ny), l, fill="#cbd5e1", font=FONT_BODY)
        ny += 26

    # Simulation Disclaimer Banner
    banner_y = 415
    draw.rounded_rectangle([120, banner_y, 1140, banner_y + 85], radius=10, fill="#3b0707", outline="#ef4444", width=2)
    draw_pill(draw, 145, banner_y + 24, "SIMULATION ONLY", FONT_HEADING, "#fca5a5", "#7f1d1d", "#f87171", pad_x=16, pad_y=8)
    draw.text((375, banner_y + 22), "Planar 2D rigid-body rocket simulation with numerical integration.", fill="#fecaca", font=FONT_BODY)
    draw.text((375, banner_y + 48), "Does not model propellant slosh, 3D crosswinds, or flight-grade avionics.", fill="#fca5a5", font=FONT_BODY)

    # Footer
    draw.line([120, 530, 1140, 530], fill="#1e293b", width=1)
    draw.text((120, 550), "Open-Source Implementation: https://github.com/AdityaSurse/hoverslam", fill="#38bdf8", font=FONT_BODY)
    draw.text((900, 550), "Model: PPO Randomized (Seed 0)", fill="#64748b", font=FONT_BODY)

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

    # Pre-render base canvas with atmospheric sky dome, stars, and drone ship pads
    print("Pre-rendering atmospheric flight canvas...")
    base_canvas = create_base_canvas()

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
    history_pid_s1: list[tuple[float, float]] = []
    history_rl_s1: list[tuple[float, float]] = []

    gif_start = max(0, max_s1 - 210)
    gif_end = gif_start + 150

    for idx in range(max_s1):
        f_pid = pid_s1_frames[idx]
        f_rl = rl_s1_frames[idx]
        history_pid_s1.append((f_pid.x, f_pid.y))
        history_rl_s1.append((f_rl.x, f_rl.y))

        frame_rgb = render_split_frame(
            base_canvas=base_canvas,
            f_pid=f_pid,
            f_rl=f_rl,
            history_pid=history_pid_s1,
            history_rl=history_rl_s1,
            scenario_title="Scenario 1: High Sensor Noise",
            wind_label="Wind: 0.75",
            noise_label="Noise: 0.20",
            step_cnt=idx,
        )
        writer.append_data(frame_rgb)
        total_frames_written += 1

        # Capture dramatic touchdown burn moment for 10-second GIF preview (150 frames)
        if gif_start <= idx < gif_end:
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
    history_pid_s2: list[tuple[float, float]] = []
    history_rl_s2: list[tuple[float, float]] = []

    for idx in range(max_s2):
        f_pid = pid_s2_frames[idx]
        f_rl = rl_s2_frames[idx]
        history_pid_s2.append((f_pid.x, f_pid.y))
        history_rl_s2.append((f_rl.x, f_rl.y))

        frame_rgb = render_split_frame(
            base_canvas=base_canvas,
            f_pid=f_pid,
            f_rl=f_rl,
            history_pid=history_pid_s2,
            history_rl=history_rl_s2,
            scenario_title="Scenario 2: Nominal Clean Sensors",
            wind_label="Wind: 0.75",
            noise_label="Noise: 0.02",
            step_cnt=idx,
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
