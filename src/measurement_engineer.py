import numpy as np
import pandas as pd
import math


def pixel_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)


def ellipse_circumference(width, depth):
    a = width / 2
    b = depth / 2
    return 2 * math.pi * math.sqrt((a**2 + b**2) / 2)


def hip_circumference(width, depth):
    a = width / 2
    b = depth / 2
    return math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))


def calculate_measurements(front_keypoints, side_keypoints, real_height_cm=170):

    # ── Scale ─────────────────────────────────────────────────────
    pixel_height = pixel_distance(
        front_keypoints["top_head"],
        front_keypoints["heel"]
    )
    if pixel_height < 100:
        raise ValueError(f"Pixel height too small ({pixel_height:.1f}px) — check head/heel points")

    scale = real_height_cm / pixel_height
    print(f"  [scale] 1px = {round(scale,4)} cm | pixel height = {round(pixel_height)}px")

    # ── Shoulder width (kept for reference) ───────────────────────
    shoulder_width_cm = pixel_distance(
        front_keypoints["left_shoulder"],
        front_keypoints["right_shoulder"]
    ) * scale

    # ── Chest: use actual chest boundary width, not shoulder ───────
    # Try chest boundary first, fall back to shoulder if not available
    if "chest_left" in front_keypoints and front_keypoints["chest_left"] is not None:
        chest_width_cm = pixel_distance(
            front_keypoints["chest_left"],
            front_keypoints["chest_right"]
        ) * scale
    else:
        # Fallback: chest width is ~85% of shoulder width
        chest_width_cm = shoulder_width_cm * 0.85
        print("  [!] No chest boundary — estimating as 85% of shoulder width")

    chest_depth_cm = pixel_distance(
        side_keypoints["chest_front"],
        side_keypoints["chest_back"]
    ) * scale

    # ── Hip width and depth ────────────────────────────────────────
    hip_width_cm = pixel_distance(
        front_keypoints["left_hip"],
        front_keypoints["right_hip"]
    ) * scale

    hip_depth_cm = pixel_distance(
        side_keypoints["hip_front"],
        side_keypoints["hip_back"]
    ) * scale

    # ── Waist: measure directly if available, else estimate ────────
    if ("waist_left" in front_keypoints and front_keypoints["waist_left"] is not None
            and "waist_front" in side_keypoints and side_keypoints["waist_front"] is not None):
        waist_width_cm = pixel_distance(
            front_keypoints["waist_left"],
            front_keypoints["waist_right"]
        ) * scale
        waist_depth_cm = pixel_distance(
            side_keypoints["waist_front"],
            side_keypoints["waist_back"]
        ) * scale
        waist_cm = ellipse_circumference(waist_width_cm, waist_depth_cm)
    else:
        # Fallback: waist is between chest and hip, slightly narrower
        chest_cm_temp = ellipse_circumference(chest_width_cm, chest_depth_cm)
        hip_cm_temp   = hip_circumference(hip_width_cm, hip_depth_cm)
        waist_cm      = ((chest_cm_temp + hip_cm_temp) / 2) * 0.85
        print("  [!] No waist boundary — estimating from chest and hip")

    # ── Final calculations ─────────────────────────────────────────
    chest_cm = ellipse_circumference(chest_width_cm, chest_depth_cm)
    hip_cm   = hip_circumference(hip_width_cm, hip_depth_cm)

    results = {
        "Shoulder Width" : round(shoulder_width_cm, 2),
        "Chest"          : round(chest_cm,          2),
        "Waist"          : round(waist_cm,           2),
        "Hips"           : round(hip_cm,             2),
    }
    return results


# Example run
if __name__ == "__main__":

    front_keypoints = {
        "left_shoulder"  : (120, 200),
        "right_shoulder" : (280, 200),
        "left_hip"       : (140, 420),
        "right_hip"      : (260, 420),
        "chest_left"     : (130, 250),   # actual chest boundary
        "chest_right"    : (270, 250),
        "waist_left"     : (145, 340),   # actual waist boundary
        "waist_right"    : (255, 340),
        "top_head"       : (200,  50),
        "heel"           : (200, 650),
    }

    side_keypoints = {
        "chest_front" : (210, 220),
        "chest_back"  : (260, 220),
        "waist_front" : (212, 340),
        "waist_back"  : (258, 340),
        "hip_front"   : (215, 420),
        "hip_back"    : (255, 420),
    }

    output = calculate_measurements(
        front_keypoints,
        side_keypoints,
        real_height_cm=170
    )

    df = pd.DataFrame(output.items(), columns=["Measurement", "Value (cm)"])
    print(df)