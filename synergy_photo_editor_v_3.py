#!/usr/bin/env python3
"""
Synergy Photo Editor – OTA Optimizer (v3.9)

Stability tweaks + tests + non‑tech friendly onboarding + **robust auto‑leveling fix**
- **Zero-arg safety**: Running with **no arguments** launches the UI when available; if Gradio is missing, we **gracefully fall back** to a friendly first‑run guide + self‑tests (exit 0).
- **Strict CLI preserved**: With any CLI args present, we still require `--input` unless you opt in via `--fallback-ui` or `SYNERGY_FALLBACK_UI=1`.
- **UI availability check**: `run_ui()` returns **True/False** (no hard exits). You can simulate no‑Gradio using `SYNERGY_FORCE_NO_GRADIO=1`.
- **Doctor mode**: `--doctor` prints a simple health check (Python, OpenCV, Pillow, write access).
- **Safer CLI**: Friendly error if `--input` path doesn’t exist; creates output folder automatically.
- **stderr safety**: Using `sys.stderr.write()` for non-stdout messages.
- **Auto‑leveling improved**: New robust angle estimator based on **HoughLinesP with length‑weighted median** and a **structure‑tensor fallback**, fixing the test failure where tilt increased after leveling.
- **One‑click launchers (NEW)**: `--install-shortcuts` creates **Launch Synergy UI.bat** (Windows) and **Launch Synergy UI.command** (macOS) so you can double‑click to open the UI.

Quick start
-----------
# 1) No args → UI (or fallback help if Gradio missing)
python synergy_photo_editor.py

# 2) Explicit UI
python synergy_photo_editor.py --ui

# 3) CLI (strict)
python synergy_photo_editor.py -i ./photos --ota airbnb --auto-level --auto-verticals --auto-crop

# 4) Opt-in fallback to UI if --input is missing
python synergy_photo_editor.py --fallback-ui
# or via env var
SYNERGY_FALLBACK_UI=1 python synergy_photo_editor.py

# 5) Simulate missing Gradio (for tests / CI)
SYNERGY_FORCE_NO_GRADIO=1 python synergy_photo_editor.py --ui

# 6) Check your setup quickly
python synergy_photo_editor.py --doctor
"""
from __future__ import annotations
import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import cv2
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# ------------------------------
# OTA export presets
# ------------------------------
OTA_PROFILES: Dict[str, Dict] = {
    "airbnb":  {"target_long": 2048, "quality": 90, "aspect": None},
    "booking": {"target_long": 2560, "quality": 90, "aspect": None},
    "vrbo":    {"target_long": 3840, "quality": 85, "aspect": None},
}

# ------------------------------
# Room‑Type presets (deltas)
# ------------------------------
TYPE_PRESETS: Dict[str, Dict[str, float]] = {
    "bedroom":  {"ev": 0.2, "contrast": -5, "shadows": 10, "highlights": -10, "clarity": 0.04},
    "bathroom": {"temp": -200, "tint": -2, "highlights": -25, "whites": -8, "clarity": 0.06},
    "kitchen":  {"temp": -150, "tint": -3, "contrast": 8, "clarity": 0.10, "dehaze": 0.04},
    "living":   {"contrast": 6, "shadows": 8, "clarity": 0.08},
    "exterior": {"contrast": 10, "dehaze": 0.08, "clarity": 0.10, "highlights": -15},
    "twilight": {"temp": 300, "tint": 3, "highlights": -30, "shadows": 12, "contrast": 6},
}

DEFAULT_TYPE_KEYWORDS: Dict[str, str] = {
    "bed": "bedroom", "br": "bedroom", "primary": "bedroom",
    "bath": "bathroom", "vanity": "bathroom",
    "kitchen": "kitchen", "kt": "kitchen",
    "living": "living", "sofa": "living", "lounge": "living",
    "exterior": "exterior", "front": "exterior", "patio": "exterior", "balcony": "exterior",
    "twilight": "twilight", "dusk": "twilight",
}

# ------------------------------
# IO helpers
# ------------------------------

def _read_image(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.clip(img_rgb.astype(np.float32) / 255.0, 0.0, 1.0)


def _save_jpeg(img_rgb: np.ndarray, out_path: Path, quality: int = 90, strip_exif: bool = True):
    img_uint8 = np.clip(np.rint(img_rgb * 255.0), 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    save_kwargs = {"format": "JPEG", "quality": int(quality), "optimize": True}
    if strip_exif and hasattr(pil_img, "getexif"):
        pil_img.info.pop('exif', None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(out_path), **save_kwargs)

# ------------------------------
# Geometry helpers
# ------------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def _from_uint8(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def _warp_affine_with_mask(img: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    warped = cv2.warpAffine(_to_uint8(img), M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpAffine(mask, M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _from_uint8(warped), warped_mask


def _auto_crop_from_mask(img: np.ndarray, mask: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    x1 = max(0, x1 + pad)
    y1 = max(0, y1 + pad)
    x2 = min(mask.shape[1]-1, x2 - pad)
    y2 = min(mask.shape[0]-1, y2 - pad)
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2+1, x1:x2+1]

# ------------------------------
# Auto‑Level & Auto‑Verticals (robust)
# ------------------------------

def _normalize_angle_deg(a: float) -> float:
    """Map any angle to (-90, 90] degrees for horizontal-dominant analysis."""
    while a <= -90.0:
        a += 180.0
    while a > 90.0:
        a -= 180.0
    return a


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * cw[-1]
    idx = int(np.searchsorted(cw, cutoff))
    return float(v[min(idx, len(v)-1)])


def _estimate_level_angle(img: np.ndarray, max_abs_angle: float = 10.0) -> float:
    """Estimate horizon tilt in degrees.

    Strategy:
    1) Detect line segments via HoughLinesP.
    2) Keep segments that are **near-horizontal** (|angle| <= 45° after normalization).
    3) Compute a **length‑weighted median** of their angles.
    4) Fallback to a **structure‑tensor** orientation if no reliable lines.
    5) Return the corrective rotation (negative of measured median), clamped.
    """
    g = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    edges = cv2.Canny(g, 60, 180, L2gradient=True)

    h, w = g.shape[:2]
    min_len = max(10, min(h, w) // 6)
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=min_len, maxLineGap=20)

    angles: List[float] = []
    weights: List[float] = []
    if linesP is not None and len(linesP) > 0:
        for x1, y1, x2, y2 in linesP[:, 0]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = max(1.0, np.hypot(dx, dy))
            ang = np.degrees(np.arctan2(dy, dx))  # -180..180, 0 is horizontal
            ang = _normalize_angle_deg(ang)
            if abs(ang) <= 45.0:  # near-horizontal only
                angles.append(ang)
                weights.append(length)

    if len(angles) >= 5:
        med = _weighted_median(np.array(angles, dtype=np.float32), np.array(weights, dtype=np.float32))
        angle = float(np.clip(-med, -max_abs_angle, max_abs_angle))
        if abs(angle) < 0.1:
            return 0.0
        return angle

    # Fallback: structure tensor orientation (robust for textures without long lines)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Gxx = cv2.GaussianBlur(gx * gx, (0, 0), 3)
    Gyy = cv2.GaussianBlur(gy * gy, (0, 0), 3)
    Gxy = cv2.GaussianBlur(gx * gy, (0, 0), 3)
    denom = (Gxx - Gyy)
    theta = 0.5 * np.arctan2(2.0 * Gxy.mean(), denom.mean() + 1e-9)  # orientation of dominant gradient
    ang_deg = np.degrees(theta)  # 0 is horizontal
    ang_deg = _normalize_angle_deg(ang_deg)
    angle = float(np.clip(-ang_deg, -max_abs_angle, max_abs_angle))
    if abs(angle) < 0.1:
        return 0.0
    return angle


def auto_level(img: np.ndarray, max_abs_angle: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    angle = _estimate_level_angle(img, max_abs_angle=max_abs_angle)
    if angle == 0.0:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return img, mask
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    out, mask = _warp_affine_with_mask(img, M, w, h)
    return out, mask


def _estimate_vertical_shear(img: np.ndarray, max_abs_deg: float = 10.0) -> float:
    g = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    edges = cv2.Canny(g, 60, 180, L2gradient=True)
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=min(img.shape[:2])//6, maxLineGap=10)
    if linesP is None or len(linesP) < 5:
        return 0.0
    angs = []
    for x1,y1,x2,y2 in linesP[:,0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        dev_to_vert = min(abs(angle - 90), abs(angle + 90))
        if dev_to_vert < 30:
            dev = (90 - angle)
            angs.append(dev)
    if not angs:
        return 0.0
    med_dev = float(np.median(angs))
    med_dev = float(np.clip(med_dev, -max_abs_deg, max_abs_deg))
    if abs(med_dev) < 0.2:
        return 0.0
    shear = np.tan(np.radians(med_dev))
    return shear


def auto_verticals(img: np.ndarray, max_abs_deg: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    sh = _estimate_vertical_shear(img, max_abs_deg=max_abs_deg)
    if abs(sh) < 1e-4:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return img, mask
    new_w = int(round(w + abs(sh) * h))
    tx = 0 if sh >= 0 else abs(sh) * h
    M = np.array([[1.0, sh, tx], [0.0, 1.0, 0.0]], dtype=np.float32)
    out, mask = _warp_affine_with_mask(img, M, new_w, h)
    return out, mask

# ------------------------------
# Color/Tone ops
# ------------------------------

def ev_to_gain(ev: float) -> float:
    return float(2.0 ** ev)


def adjust_exposure(img: np.ndarray, ev: float) -> np.ndarray:
    if abs(ev) < 1e-6:
        return img
    gain = ev_to_gain(ev)
    return np.clip(img * gain, 0.0, 1.0)


def auto_exposure(img: np.ndarray, target_mean: float = 0.52) -> np.ndarray:
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    L = lab[..., 0]
    mean_L = float(np.mean(L))
    if mean_L < 1e-6:
        return img
    gain = np.clip(target_mean / mean_L, 0.5, 2.0)
    out = np.clip(img * gain, 0.0, 1.0)
    return out


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    means = np.mean(img, axis=(0, 1)) + 1e-6
    gray = np.mean(means)
    gains = gray / means
    out = np.clip(img * gains, 0.0, 1.0)
    return out


def temp_tint_adjust(img: np.ndarray, temp_K: Optional[float], tint: Optional[float]) -> np.ndarray:
    out = img.copy()
    if temp_K is not None:
        t = np.clip((temp_K - 2000.0) / (9000.0 - 2000.0), 0.0, 1.0)
        r_gain = 0.9 + 0.6 * t
        b_gain = 1.5 - 0.6 * t
        gains = np.array([r_gain, 1.0, b_gain], dtype=np.float32)
        out = np.clip(out * gains, 0.0, 1.0)
    if tint is not None:
        tint = float(np.clip(tint, -30, 30)) / 30.0
        m = 1.0 + 0.12 * tint
        g = 1.0 - 0.12 * tint
        gains = np.array([m, g, m], dtype=np.float32)
        out = np.clip(out * gains, 0.0, 1.0)
    return out


def adjust_contrast(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 1e-6:
        return img
    a = np.clip(amount, -100.0, 100.0) / 100.0
    mid = 0.5
    out = np.clip((img - mid) * (1 + 0.9 * a) + mid, 0.0, 1.0)
    return out


def local_contrast(img: np.ndarray, strength: float = 0.15, radius: int = 21) -> np.ndarray:
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    L = lab[..., 0]
    blur = cv2.GaussianBlur(L, (0, 0), radius / 3.0)
    detail = L - blur
    L2 = np.clip(L + strength * detail, 0.0, 1.0)
    lab[..., 0] = L2
    out = cv2.cvtColor((lab * 255).astype(np.uint8), cv2.COLOR_Lab2RGB).astype(np.float32) / 255.0
    return out


def highlights_shadows(img: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    L = lab[..., 0]
    h = np.clip(-highlights / 100.0, 0.0, 1.0)
    s = np.clip(shadows / 100.0, 0.0, 1.0)
    highlight_mask = np.clip((L - 0.6) / 0.4, 0.0, 1.0)
    shadow_mask = np.clip((0.4 - L) / 0.4, 0.0, 1.0)
    L2 = L.copy()
    if h > 0:
        L2 = L2 - h * highlight_mask * (L - 0.6) * 0.8
    if s > 0:
        L2 = L2 + s * shadow_mask * (0.6 - L) * 0.8
    L2 = np.clip(L2, 0.0, 1.0)
    lab[..., 0] = L2
    out = cv2.cvtColor((lab * 255).astype(np.uint8), cv2.COLOR_Lab2RGB).astype(np.float32) / 255.0
    return out


def whites_blacks(img: np.ndarray, whites: float, blacks: float) -> np.ndarray:
    w = np.clip(whites, -100.0, 100.0) / 100.0
    b = np.clip(blacks, -100.0, 100.0) / 100.0
    x = img
    if abs(b) > 1e-6:
        x = np.clip(x + b * (0.08 - x) * (x < 0.2), 0.0, 1.0)
    if abs(w) > 1e-6:
        x = np.clip(x + w * (x - 0.92) * (x > 0.8), 0.0, 1.0)
    return x


def unsharp(img: np.ndarray, radius: int = 1, amount: float = 0.8, threshold: float = 0.0) -> np.ndarray:
    sigma = max(radius, 1) / 3.0
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    detail = img - blur
    if threshold > 0:
        mask = (np.abs(detail).mean(axis=2, keepdims=True) > threshold).astype(np.float32)
        detail *= mask
    out = np.clip(img + amount * detail, 0.0, 1.0)
    return out


def dehaze_lite(img: np.ndarray, strength: float = 0.1) -> np.ndarray:
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(1.0, 2.0 * strength + 1.0), tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_Lab2RGB).astype(np.float32) / 255.0
    return out

# ------------------------------
# Resize helpers
# ------------------------------

def resize_long_edge(img: np.ndarray, target_long: Optional[int]) -> np.ndarray:
    if not target_long or target_long <= 0:
        return img
    h, w = img.shape[:2]
    long = max(h, w)
    if long == target_long:
        return img
    scale = target_long / float(long)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

# ------------------------------
# Presets & param merge
# ------------------------------

BASE_PRESET = {
    "temp": 5000.0,
    "tint": 5.0,
    "ev": 0.8,
    "contrast": 10.0,
    "highlights": -30.0,
    "shadows": 35.0,
    "whites": -10.0,
    "blacks": -5.0,
    "clarity": 0.12,
    "dehaze": 0.05,
    "sharpen_amount": 0.5,
    "sharpen_radius": 1,
}


def apply_type_preset(params: Dict[str, float], room_type: Optional[str]) -> Dict[str, float]:
    out = dict(params)
    if room_type and room_type in TYPE_PRESETS:
        for k, v in TYPE_PRESETS[room_type].items():
            if k in out and isinstance(out[k], (int, float)):
                out[k] = out[k] + v
            else:
                out[k] = v
    return out

# ------------------------------
# Pipeline
# ------------------------------

def process_image(
    img: np.ndarray,
    *,
    auto_leveling: bool = False,
    max_level_angle: float = 10.0,
    auto_verticals_on: bool = False,
    max_vertical_angle: float = 10.0,
    auto_crop: bool = False,
    temp: Optional[float] = None,
    tint: Optional[float] = None,
    ev: Optional[float] = None,
    auto_ev: bool = False,
    contrast: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    whites: float = 0.0,
    blacks: float = 0.0,
    clarity: float = 0.0,
    dehaze: float = 0.0,
    sharpen_amount: float = 0.0,
    sharpen_radius: int = 1,
    target_long_edge: Optional[int] = None,
) -> np.ndarray:
    masks: List[np.ndarray] = []
    x = img.copy()

    # geometry
    if auto_leveling:
        x, m = auto_level(x, max_abs_angle=max_level_angle)
        masks.append(m)
    if auto_verticals_on:
        x, m = auto_verticals(x, max_abs_deg=max_vertical_angle)
        masks.append(m)
    if auto_crop and masks:
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_and(mask, m)
        x = _auto_crop_from_mask(x, mask, pad=2)

    # color/tone
    x = gray_world_white_balance(x)
    x = temp_tint_adjust(x, temp_K=temp, tint=tint)
    if auto_ev:
        x = auto_exposure(x)
    if ev is not None:
        x = adjust_exposure(x, ev)
    if abs(contrast) > 1e-6:
        x = adjust_contrast(x, contrast)
    if abs(highlights) > 1e-6 or abs(shadows) > 1e-6:
        x = highlights_shadows(x, highlights=highlights, shadows=shadows)
    if abs(whites) > 1e-6 or abs(blacks) > 1e-6:
        x = whites_blacks(x, whites=whites, blacks=blacks)
    if clarity and abs(clarity) > 1e-6:
        x = local_contrast(x, strength=float(clarity))
    if dehaze and abs(dehaze) > 1e-6:
        x = dehaze_lite(x, strength=float(dehaze))
    if sharpen_amount and sharpen_amount > 0:
        x = unsharp(x, radius=sharpen_radius, amount=sharpen_amount, threshold=0.0)

    # resize
    x = resize_long_edge(x, target_long_edge)
    return x

# ------------------------------
# CLI
# ------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Synergy Photo Editor – OTA Optimizer (v3.9)")
    p.add_argument('--ui', action='store_true', help='Launch visual editor (Gradio)')
    p.add_argument('--fallback-ui', action='store_true', help='If CLI is missing --input, launch UI instead of exiting (or set SYNERGY_FALLBACK_UI=1)')
    p.add_argument('--selftest', action='store_true', help='Run built-in tests and exit')
    p.add_argument('--doctor', action='store_true', help='Run environment health check and exit')
    p.add_argument('--install-shortcuts', action='store_true', help='Create double-click launchers for Windows/macOS in the current folder')
    p.add_argument('--input', '-i', required=False, help='Input file or folder of images (PNG/JPEG)')
    p.add_argument('--output', '-o', default='output', help='Output folder (auto-created)')

    p.add_argument('--auto-level', action='store_true', help='Auto-straighten horizon')
    p.add_argument('--max-level-angle', type=float, default=10.0, help='Max rotation correction (deg)')
    p.add_argument('--auto-verticals', action='store_true', help='Correct leaning verticals (horizontal shear)')
    p.add_argument('--max-vertical-angle', type=float, default=10.0, help='Max vertical correction (deg)')
    p.add_argument('--auto-crop', action='store_true', help='Crop empty borders after transforms')

    p.add_argument('--ota', choices=sorted(OTA_PROFILES.keys()), help='OTA export preset')
    p.add_argument('--room-type', choices=sorted(TYPE_PRESETS.keys()), help='Room type preset to apply')
    p.add_argument('--batch-types', action='store_true', help='Infer room type from filename keywords')

    p.add_argument('--target-width', type=int, help='Manual long-edge size (overrides OTA preset)')
    p.add_argument('--quality', type=int, default=None, help='JPEG quality 60..100 (overrides OTA preset)')
    p.add_argument('--keep-exif', action='store_true', help='Keep EXIF metadata')
    return p


# ------------------------------
# Shortcut installers (double-click launchers)
# ------------------------------

def install_shortcuts(target_dir: Path, python_cmd: str = "python") -> List[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).name
    created: List[Path] = []

    # Windows .bat
    bat = target_dir / "Launch Synergy UI.bat"
    bat.write_text(
        "@echo off
"
        "cd /d %~dp0
"
        f"{python_cmd} {script} --ui
"
        "pause
",
        encoding='utf-8'
    )
    created.append(bat)

    # macOS .command
    cmd = target_dir / "Launch Synergy UI.command"
    cmd.write_text(
        "#!/bin/bash
"
        "cd \"$(dirname \"$0\")\"
"
        f"{python_cmd} {script} --ui
",
        encoding='utf-8'
    )
    try:
        os.chmod(cmd, 0o755)
    except Exception:
        pass
    created.append(cmd)

    return created

# ------------------------------
# Mode decision helpers
# ------------------------------

def _truthy(s: Optional[str]) -> bool:
    if s is None:
        return False
    return s.strip().lower() in {"1", "true", "yes", "on"}


def _should_fallback_ui(flag: bool, env_value: Optional[str]) -> bool:
    return bool(flag or _truthy(env_value))


def _decide_mode(argv: List[str], args: argparse.Namespace) -> str:
    """Return one of: 'ui', 'cli', 'fallback-ui', 'error'.
    - No args (argv length 1): 'ui'
    - --ui: 'ui'
    - Has any args and no --input: 'fallback-ui' if flag/env set, else 'error'
    - Otherwise: 'cli'
    """
    if len(argv) <= 1:
        return 'ui'
    if getattr(args, 'selftest', False):
        return 'cli'  # handled earlier; just a placeholder
    if getattr(args, 'ui', False):
        return 'ui'
    if not getattr(args, 'input', None):
        return 'fallback-ui' if _should_fallback_ui(getattr(args, 'fallback_ui', False), os.environ.get('SYNERGY_FALLBACK_UI')) else 'error'
    return 'cli'


def _collect_images(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        return [input_path]
    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    files = [p for p in input_path.rglob('*') if p.suffix in exts]
    if not files:
        raise ValueError(f"No PNG/JPEG files found under {input_path}")
    return files


def _guess_room_type(name: str) -> Optional[str]:
    low = name.lower()
    for kw, t in DEFAULT_TYPE_KEYWORDS.items():
        if kw in low:
            return t
    return None


def run_cli(args: argparse.Namespace):
    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_long = None
    quality = 90
    if args.ota:
        prof = OTA_PROFILES[args.ota]
        target_long = prof.get('target_long')
        quality = prof.get('quality', quality)
    if args.target_width:
        target_long = args.target_width
    if args.quality is not None:
        quality = args.quality

    files = _collect_images(input_path)

    base = dict(BASE_PRESET)
    global_type = args.room_type if args.room_type else None

    processed = 0
    for f in tqdm(files, desc="Processing"):
        try:
            img = _read_image(f)
            params = dict(base)
            rt = global_type
            if args.batch_types and not rt:
                rt = _guess_room_type(f.name)
            params = apply_type_preset(params, rt) if rt else params

            out = process_image(
                img,
                auto_leveling=args.auto_level,
                max_level_angle=args.max_level_angle,
                auto_verticals_on=args.auto_verticals,
                max_vertical_angle=args.max_vertical_angle,
                auto_crop=args.auto_crop,
                temp=params.get('temp', None),
                tint=params.get('tint', None),
                ev=params.get('ev', None),
                auto_ev=False,
                contrast=params.get('contrast', 0.0),
                highlights=params.get('highlights', 0.0),
                shadows=params.get('shadows', 0.0),
                whites=params.get('whites', 0.0),
                blacks=params.get('blacks', 0.0),
                clarity=params.get('clarity', 0.0),
                dehaze=params.get('dehaze', 0.0),
                sharpen_amount=params.get('sharpen_amount', 0.0),
                sharpen_radius=params.get('sharpen_radius', 1),
                target_long_edge=target_long,
            )
            rel = f.name if input_path.is_file() else f.relative_to(input_path)
            out_path = out_dir / Path(rel).with_suffix('.jpg')
            _save_jpeg(out, out_path, quality=quality, strip_exif=(not args.keep_exif))
            processed += 1
        except Exception as e:
            print(f"[WARN] {f}: {e}")

    print(f"Done. Processed {processed} file(s). Outputs → {out_dir.resolve()}")

# ------------------------------
# Gradio UI (returns bool success)
# ------------------------------

def run_ui() -> bool:  # pragma: no cover
    # Allow tests/CI to simulate a no-Gradio environment
    if _truthy(os.environ.get('SYNERGY_FORCE_NO_GRADIO')):
        sys.stderr.write("[INFO] Gradio disabled via SYNERGY_FORCE_NO_GRADIO=1.\n")
        return False
    try:
        import gradio as gr
    except Exception:
        sys.stderr.write("[WARN] Gradio is not installed. Run: pip install gradio\n")
        return False

    def _single(pil_img: Image.Image,
                ota: str, room_type: str,
                auto_leveling: bool, max_level_angle: float,
                auto_verticals: bool, max_vertical_angle: float,
                auto_crop: bool,
                quality: int):
        img = np.asarray(pil_img.convert('RGB')).astype(np.float32) / 255.0
        params = apply_type_preset(dict(BASE_PRESET), room_type if room_type else None)
        prof = OTA_PROFILES.get(ota) if ota else None
        target_long = prof.get('target_long') if prof else 2048
        out = process_image(
            img,
            auto_leveling=auto_leveling,
            max_level_angle=max_level_angle,
            auto_verticals_on=auto_verticals,
            max_vertical_angle=max_vertical_angle,
            auto_crop=auto_crop,
            temp=params.get('temp'), tint=params.get('tint'), ev=params.get('ev'),
            contrast=params.get('contrast', 0), highlights=params.get('highlights', 0),
            shadows=params.get('shadows', 0), whites=params.get('whites', 0), blacks=params.get('blacks', 0),
            clarity=params.get('clarity', 0), dehaze=params.get('dehaze', 0),
            sharpen_amount=params.get('sharpen_amount', 0), sharpen_radius=params.get('sharpen_radius', 1),
            target_long_edge=target_long,
        )
        return Image.fromarray(_to_uint8(out))

    def _batch(files: List['gr.File'], ota: str, infer_types: bool, room_type: str,
               auto_leveling: bool, max_level_angle: float,
               auto_verticals: bool, max_vertical_angle: float,
               auto_crop: bool,
               quality: int):
        prof = OTA_PROFILES.get(ota) if ota else None
        target_long = prof.get('target_long') if prof else 2048
        zbuf = io.BytesIO()
        zf = zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED)
        for f in files or []:
            # Accept plain file paths from Gradio (type="filepath") or legacy gr.File objects.
            if isinstance(f, (str, Path)):
                path = Path(f)
            else:
                # Fallback: f may be a gr.File-like object with a `.name` attribute.
                path = Path(getattr(f, 'name', f))
            img = _read_image(path)
            rt = None
            if infer_types:
                rt = _guess_room_type(path.name)
            if not rt:
                rt = room_type if room_type else None
            params = apply_type_preset(dict(BASE_PRESET), rt)
            out = process_image(
                img,
                auto_leveling=auto_leveling,
                max_level_angle=max_level_angle,
                auto_verticals_on=auto_verticals,
                max_vertical_angle=max_vertical_angle,
                auto_crop=auto_crop,
                temp=params.get('temp'), tint=params.get('tint'), ev=params.get('ev'),
                contrast=params.get('contrast', 0), highlights=params.get('highlights', 0),
                shadows=params.get('shadows', 0), whites=params.get('whites', 0), blacks=params.get('blacks', 0),
                clarity=params.get('clarity', 0), dehaze=params.get('dehaze', 0),
                sharpen_amount=params.get('sharpen_amount', 0), sharpen_radius=params.get('sharpen_radius', 1),
                target_long_edge=target_long,
            )
            out_img = Image.fromarray(_to_uint8(out))
            out_name = path.with_suffix('.jpg').name
            bio = io.BytesIO()
            out_img.save(bio, format='JPEG', quality=prof.get('quality', 90) if prof else 90, optimize=True)
            zf.writestr(out_name, bio.getvalue())
        zf.close()
        zbuf.seek(0)
        try:
            zbuf.name = 'synergy_edited.zip'
        except Exception:
            pass
        return zbuf

    with gr.Blocks(title="Synergy Photo Editor v3.8") as demo:
        gr.Markdown("## Synergy Photo Editor v3.8 — OTA Optimizer (Room‑Type + UI)")
        with gr.Tabs():
            with gr.Tab("Single"):
                in_img = gr.Image(type='pil', label='Input Image')
                with gr.Row():
                    ota = gr.Dropdown(choices=['', *sorted(OTA_PROFILES.keys())], value='airbnb', label='OTA Preset')
                    room_type = gr.Dropdown(choices=['', *sorted(TYPE_PRESETS.keys())], value='', label='Room Type Preset')
                with gr.Row():
                    auto_leveling = gr.Checkbox(value=True, label='Auto Level')
                    max_level_angle = gr.Slider(0, 20, value=10, step=0.5, label='Max Level (°)')
                    auto_verticals = gr.Checkbox(value=True, label='Auto Verticals')
                    max_vertical_angle = gr.Slider(0, 20, value=10, step=0.5, label='Max Verticals (°)')
                    auto_crop = gr.Checkbox(value=True, label='Auto Crop')
                quality = gr.Slider(60, 100, value=90, step=1, label='JPEG Quality')
                btn_single = gr.Button('Process Single')
                out_img = gr.Image(type='pil', label='Output')
                btn_single.click(_single,
                                 inputs=[in_img, ota, room_type, auto_leveling, max_level_angle, auto_verticals, max_vertical_angle, auto_crop, quality],
                                 outputs=[out_img])
            with gr.Tab("Batch"):
                files = gr.Files(label='Upload PNG/JPEGs', file_count='multiple', type='filepath')
                ota_b = gr.Dropdown(choices=['', *sorted(OTA_PROFILES.keys())], value='airbnb', label='OTA Preset')
                with gr.Row():
                    infer_types = gr.Checkbox(value=True, label='Infer room type from filename')
                    room_type_b = gr.Dropdown(choices=['', *sorted(TYPE_PRESETS.keys())], value='', label='Fallback Room Type')
                with gr.Row():
                    auto_leveling_b = gr.Checkbox(value=True, label='Auto Level')
                    max_level_angle_b = gr.Slider(0, 20, value=10, step=0.5, label='Max Level (°)')
                    auto_verticals_b = gr.Checkbox(value=True, label='Auto Verticals')
                    max_vertical_angle_b = gr.Slider(0, 20, value=10, step=0.5, label='Max Verticals (°)')
                    auto_crop_b = gr.Checkbox(value=True, label='Auto Crop')
                quality_b = gr.Slider(60, 100, value=90, step=1, label='JPEG Quality')
                btn_batch = gr.Button('Process Batch (download ZIP)')
                zip_out = gr.File(label='Download ZIP')
                btn_batch.click(_batch,
                                inputs=[files, ota_b, infer_types, room_type_b, auto_leveling_b, max_level_angle_b, auto_verticals_b, max_vertical_angle_b, auto_crop_b, quality_b],
                                outputs=[zip_out])
        demo.launch()
    return True

# ------------------------------
# Tests
# ------------------------------

def _make_striped_image(w=640, h=480, horizontal=True) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    step = 20
    color1 = (200, 200, 200)
    color2 = (40, 40, 40)
    for i in range(0, h if horizontal else w, step):
        if horizontal:
            img[i:i+step//2, :, :] = color1
            img[i+step//2:i+step, :, :] = color2
        else:
            img[:, i:i+step//2, :] = color1
            img[:, i+step//2:i+step, :] = color2
    return img.astype(np.float32) / 255.0


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _tmp_dir() -> Path:
    import tempfile
    p = Path(tempfile.mkdtemp(prefix="synphoto_"))
    return p


def run_selftest() -> int:
    # Image-processing tests
    base = _make_striped_image()
    tilted = _rotate_image(base, 5.0)
    before = abs(_estimate_level_angle(tilted))
    fixed, _ = auto_level(tilted)
    after = abs(_estimate_level_angle(fixed))
    assert after <= before, f"auto_level failed: after={after:.3f} > before={before:.3f}"

    # Additional robustness tests for auto_level
    tilted_neg = _rotate_image(base, -7.0)
    before_n = abs(_estimate_level_angle(tilted_neg))
    fixed_n, _ = auto_level(tilted_neg)
    after_n = abs(_estimate_level_angle(fixed_n))
    assert after_n <= before_n + 1e-3, f"auto_level(-) failed: after={after_n:.3f} > before={before_n:.3f}"

    flat = base.copy()
    zero = abs(_estimate_level_angle(flat))
    assert zero <= 0.5, f"auto_level zero-tilt estimate too high: {zero:.3f}"

    vertical = _make_striped_image(horizontal=False)
    h, w = vertical.shape[:2]
    sh = np.tan(np.radians(5.0))
    M = np.array([[1.0, sh, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    sheared = cv2.warpAffine(vertical, M, (int(round(w + abs(sh)*h)), h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sv_fixed, _ = auto_verticals(sheared)
    est_sh_after = abs(_estimate_vertical_shear(sv_fixed))
    est_sh_before = abs(_estimate_vertical_shear(sheared))
    assert est_sh_after <= est_sh_before, f"auto_verticals failed: after={est_sh_after:.4f} > before={est_sh_before:.4f}"

    img = np.zeros((300, 500, 3), np.float32)
    out = resize_long_edge(img, 1000)
    assert max(out.shape[:2]) == 1000, "resize_long_edge failed to set long edge to target"

    # Decision-logic tests
    parser = build_arg_parser()
    # Case A: no args → UI
    a, _ = parser.parse_known_args([])
    assert _decide_mode(["prog"], a) == 'ui'
    # Case B: explicit --ui → UI
    b, _ = parser.parse_known_args(["--ui"])
    assert _decide_mode(["prog", "--ui"], b) == 'ui'
    # Case C: CLI missing --input, no fallback → error
    c, _ = parser.parse_known_args(["--auto-level"])  # any arg present counts as CLI attempt
    assert _decide_mode(["prog", "--auto-level"], c) == 'error'
    # Case D: CLI missing --input, with fallback flag → fallback-ui
    d, _ = parser.parse_known_args(["--fallback-ui"])
    assert _decide_mode(["prog", "--fallback-ui"], d) == 'fallback-ui'

    # Case E: simulate missing Gradio → run_ui returns False
    os.environ['SYNERGY_FORCE_NO_GRADIO'] = '1'
    try:
        assert run_ui() is False
    finally:
        os.environ.pop('SYNERGY_FORCE_NO_GRADIO', None)

    # Filesystem edge cases for _collect_images
    tmp = _tmp_dir()
    try:
        # 1) Non-existent path
        try:
            _collect_images(tmp / "nope")
            assert False, "_collect_images should raise on missing path"
        except FileNotFoundError:
            pass
        # 2) Empty dir
        try:
            _collect_images(tmp)
            assert False, "_collect_images should raise on empty folder"
        except ValueError:
            pass
        # 3) Single file
        pimg = tmp / "x.png"
        cv2.imwrite(str(pimg), (_to_uint8(np.zeros((10, 10, 3), np.float32))[:, :, ::-1]))
        got = _collect_images(pimg)
        assert len(got) == 1 and got[0] == pimg
        # 4) Shortcut installer dry run
        created = install_shortcuts(tmp)
        names = {p.name for p in created}
        assert "Launch Synergy UI.bat" in names and "Launch Synergy UI.command" in names
    finally:
        pass

    print("Selftests passed ✓")
    return 0

# ------------------------------
# Entry + First‑run helper
# ------------------------------

def _first_run_help(parser: argparse.ArgumentParser):
    sys.stderr.write("\n=== Synergy Photo Editor – First Run Guide ===\n")
    sys.stderr.write("If the visual editor didn't open, install it with:\n    pip install gradio\n\n")
    sys.stderr.write("Quick ways to start:\n")
    sys.stderr.write("  • Visual editor:  python synergy_photo_editor.py --ui\n")
    sys.stderr.write("  • Batch folder:   python synergy_photo_editor.py -i ./photos --ota airbnb --auto-level --auto-verticals --auto-crop\n\n")
    sys.stderr.write("Need help? Run:  python synergy_photo_editor.py --doctor\n\n")
    sys.stderr.write(parser.format_help())

if __name__ == '__main__':
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    if getattr(args, 'selftest', False):
        rc = run_selftest()
        sys.exit(rc)

    if getattr(args, 'doctor', False):
        rc = 0
        try:
            sys.stderr.write("Checking environment...
")
            sys.stderr.write(f"Python: {sys.version.split()[0]}
")
            import importlib
            for mod in ("cv2", "PIL", "numpy", "gradio"):
                try:
                    importlib.import_module(mod)
                    sys.stderr.write(f"ok  - {mod}
")
                except Exception as e:
                    sys.stderr.write(f"warn- {mod}: {e}
")
            # write access test
            try:
                Path('output').mkdir(exist_ok=True)
                testfile = Path('output/.write_test')
                testfile.write_text('ok')
                testfile.unlink(missing_ok=True)
                sys.stderr.write("ok  - write access
")
            except Exception as e:
                sys.stderr.write(f"fail- write access: {e}
"); rc = 1
        except Exception as e:
            sys.stderr.write(f"doctor encountered an error: {e}
"); rc = 1
        sys.exit(rc)

    # Decide mode robustly
    decision = _decide_mode(sys.argv, args)
    if decision == 'ui':
        if run_ui():
            sys.exit(0)
        # UI not available → friendly fallback without error code
        sys.stderr.write("[INFO] UI unavailable. Showing CLI help and running selftests.
")
        _first_run_help(parser)
        rc = run_selftest()
        sys.exit(rc)

    if decision == 'fallback-ui':
        # Opt-in fallback wants UI; if UI can't start, surface an actionable error
        if run_ui():
            sys.exit(0)
        sys.stderr.write("error: UI fallback requested but Gradio is unavailable. Install with `pip install gradio` or provide --input for CLI.
")
        sys.exit(2)

    if decision == 'error':
        sys.stderr.write("error: --input is required in CLI mode (or pass --ui, or use --fallback-ui / SYNERGY_FALLBACK_UI=1)
")
        sys.exit(2)

    # Utilities that don't require --input
    if getattr(args, 'install_shortcuts', False):
        made = install_shortcuts(Path('.'))
        sys.stderr.write("Created launchers:
" + "
".join(f"  - {p}" for p in made) + "
")
        sys.exit(0)

    # CLI
    args = parser.parse_args()
    run_cli(args)
