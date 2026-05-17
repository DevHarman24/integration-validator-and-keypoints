# ─────────────────────────────────────────────────────────────────
# FILE: src/geometry_engineer.py
# PURPOSE: Takes keypoints from pose_engine → returns body measurements
# CONNECTS TO: src/main.py (main.py will import and call this)
# ─────────────────────────────────────────────────────────────────

import numpy as np
import math


# ── HELPER FUNCTIONS ─────────────────────────────────────────────

def euclidean_distance(p1, p2):
    """Pixel distance between two points."""
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def pixels_to_cm(px, scale):
    """Convert pixel distance to centimetres."""
    if px is None or scale is None:
        return None
    return round(px * scale, 2)


def ellipse_circumference(width_cm, depth_cm):
    """Body part circumference using ellipse formula (more accurate than circle)."""
    if width_cm is None or depth_cm is None:
        return None
    a = width_cm / 2
    b = depth_cm / 2
    return round(2 * math.pi * math.sqrt((a**2 + b**2) / 2), 2)


# ── SCALE CALCULATOR ─────────────────────────────────────────────

def get_scale(top_point, bottom_point, real_height_cm):
    """
    Returns cm-per-pixel using person's known height.
    top_point    = pixel coords of top of head [x, y]
    bottom_point = pixel coords of feet        [x, y]
    """
    if top_point is None or bottom_point is None:
        return None
    pixel_height = euclidean_distance(top_point, bottom_point)
    if pixel_height == 0 or pixel_height is None:
        return None
    return real_height_cm / pixel_height


# ── MAIN FUNCTION ─────────────────────────────────────────────────

def calculate_measurements(front_kp, side_kp, height_cm,
                            top_point=None, bottom_point=None):
    """
    MAIN FUNCTION — called by main.py after pose_engine runs.

    Input:
        front_kp     : dict — keypoints from front image
                       Must have: shoulder_left, shoulder_right,
                                  hip_left, hip_right
        side_kp      : dict — same keys, from side image
        height_cm    : float — user's real height in cm
        top_point    : [x, y] — top of head pixel (optional)
        bottom_point : [x, y] — feet pixel (optional)

    Output:
        dict  — {chest, waist, hips} in cm
        OR
        str   — 'invalid input' if keypoints are missing
    """

    # ── Validate required keypoints ──────────────────────────────
    required = ['shoulder_left', 'shoulder_right', 'hip_left', 'hip_right']
    for key in required:
        if key not in front_kp or front_kp[key] is None:
            print(f'[geometry] Missing front keypoint: {key}')
            return 'invalid input'
        if key not in side_kp or side_kp[key] is None:
            print(f'[geometry] Missing side keypoint: {key}')
            return 'invalid input'

    # ── Estimate top/bottom if not provided ──────────────────────
    if top_point is None:
        # Estimate head above shoulder midpoint
        sh_x = (front_kp['shoulder_left'][0] + front_kp['shoulder_right'][0]) / 2
        sh_y = (front_kp['shoulder_left'][1] + front_kp['shoulder_right'][1]) / 2
        top_point = [sh_x, sh_y * 0.85]

    if bottom_point is None:
        # Estimate feet below hip midpoint
        hip_x = (front_kp['hip_left'][0] + front_kp['hip_right'][0]) / 2
        hip_y = (front_kp['hip_left'][1] + front_kp['hip_right'][1]) / 2
        bottom_point = [hip_x, hip_y * 1.55]

    # ── Scale ────────────────────────────────────────────────────
    scale = get_scale(top_point, bottom_point, height_cm)
    if scale is None:
        return 'invalid input'

    # ── Pixel widths (front image) ────────────────────────────────
    sh_width_px  = euclidean_distance(front_kp['shoulder_left'],
                                       front_kp['shoulder_right'])
    hip_width_px = euclidean_distance(front_kp['hip_left'],
                                       front_kp['hip_right'])

    # ── Pixel depths (side image) ─────────────────────────────────
    sh_depth_px  = euclidean_distance(side_kp['shoulder_left'],
                                       side_kp['shoulder_right'])
    hip_depth_px = euclidean_distance(side_kp['hip_left'],
                                       side_kp['hip_right'])

    # ── Convert to cm ─────────────────────────────────────────────
    sh_w  = pixels_to_cm(sh_width_px,  scale)
    hip_w = pixels_to_cm(hip_width_px, scale)
    sh_d  = pixels_to_cm(sh_depth_px,  scale)
    hip_d = pixels_to_cm(hip_depth_px, scale)

    # ── Waist estimate (80% of avg of chest + hip) ────────────────
    waist_w = round(((sh_w + hip_w) / 2) * 0.80, 2)
    waist_d = round(((sh_d + hip_d) / 2) * 0.80, 2)

    # ── Ellipse circumferences ────────────────────────────────────
    chest = ellipse_circumference(sh_w,    sh_d)
    waist = ellipse_circumference(waist_w, waist_d)
    hips  = ellipse_circumference(hip_w,   hip_d)

    return {
        'chest': chest,
        'waist': waist,
        'hips' : hips
    }