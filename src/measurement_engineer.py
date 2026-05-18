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

    circumference = (
        2 * math.pi *
        math.sqrt((a**2 + b**2) / 2)
    )

    return circumference


def hip_circumference(width, depth):
    a = width / 2
    b = depth / 2

    circumference = math.pi * (
        3 * (a + b)
        - math.sqrt(
            (3 * a + b) *
            (a + 3 * b)
        )
    )

    return circumference


def calculate_measurements(
    front_keypoints,
    side_keypoints,
    real_height_cm=170
):

    # Height scaling
    pixel_height = pixel_distance(
        front_keypoints["top_head"],
        front_keypoints["heel"]
    )

    scale = real_height_cm / pixel_height

    # Shoulder width
    shoulder_width_pixels = pixel_distance(
        front_keypoints["left_shoulder"],
        front_keypoints["right_shoulder"]
    )

    shoulder_width_cm = (
        shoulder_width_pixels * scale
    )

    # Hip width
    hip_width_pixels = pixel_distance(
        front_keypoints["left_hip"],
        front_keypoints["right_hip"]
    )

    hip_width_cm = hip_width_pixels * scale

    # Chest depth
    chest_depth_pixels = pixel_distance(
        side_keypoints["chest_front"],
        side_keypoints["chest_back"]
    )

    chest_depth_cm = chest_depth_pixels * scale

    # Hip depth
    hip_depth_pixels = pixel_distance(
        side_keypoints["hip_front"],
        side_keypoints["hip_back"]
    )

    hip_depth_cm = hip_depth_pixels * scale

    # Final measurements
    chest_cm = ellipse_circumference(
        shoulder_width_cm,
        chest_depth_cm
    )

    hip_cm = hip_circumference(
        hip_width_cm,
        hip_depth_cm
    )

    waist_cm = (
        (chest_cm + hip_cm)
        / 2
    ) * 0.9

    results = {
        "Shoulder Width": round(
            shoulder_width_cm, 2
        ),

        "Chest": round(
            chest_cm, 2
        ),

        "Waist": round(
            waist_cm, 2
        ),

        "Hips": round(
            hip_cm, 2
        )
    }

    return results



# Example run
if __name__ == "__main__":

    front_keypoints = {
        "left_shoulder": (120, 200),
        "right_shoulder": (280, 200),

        "left_hip": (140, 420),
        "right_hip": (260, 420),

        "top_head": (200, 50),
        "heel": (200, 650)
    }

    side_keypoints = {
        "chest_front": (210, 220),
        "chest_back": (260, 220),

        "hip_front": (215, 420),
        "hip_back": (255, 420)
    }

    output = calculate_measurements(
        front_keypoints,
        side_keypoints,
        real_height_cm=170
    )

    df = pd.DataFrame(
        output.items(),
        columns=[
            "Measurement",
            "Value (cm)"
        ]
    )

    print(df)