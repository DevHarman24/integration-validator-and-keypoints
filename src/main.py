from pose_engine import PoseEngine
from validator import HumanPoseValidator
from measurement_engineer import calculate_measurements
import json
import os
import sys
import math
# pyrefly: ignore [missing-import]
import cv2


def process_image_pair(front_path, side_path, engine, validator, height_cm, output_dir):

    # Validate images
    for img_path, view in [(front_path, "front"), (side_path, "side")]:

        if not os.path.exists(img_path):
            print(f"[!] Missing file: {img_path}")
            return

        print(f"\nValidating {view} image...")

        try:
            val = validator.validate_pose(
                img_path,
                expected_view=view
            )

        except Exception as e:
            print(e)
            return

        if not val.get("valid", False):

            print("Validation failed")

            for err in val.get("errors", []):
                print(err)

            return

        print(f"{view} image OK")

    # Extract keypoints
    print("\nExtracting keypoints...")

    front_kp = engine.get_keypoints(front_path)
    side_kp = engine.get_keypoints(side_path)

    if isinstance(front_kp, str):
        print(front_kp)
        return

    if isinstance(side_kp, str):
        print(side_kp)
        return

    print(json.dumps(front_kp, indent=4))
    print(json.dumps(side_kp, indent=4))

    # FRONT MAPPING

    sh_left = front_kp.get("shoulder_left")
    sh_right = front_kp.get("shoulder_right")

    hip_left = front_kp.get("hip_left")
    hip_right = front_kp.get("hip_right")

    if not all([sh_left, sh_right, hip_left, hip_right]):
        print("Missing front landmarks")
        return

    shoulder_y = (sh_left[1] + sh_right[1]) / 2
    shoulder_x = (sh_left[0] + sh_right[0]) / 2

    hip_y = (hip_left[1] + hip_right[1]) / 2

    torso_px = abs(hip_y - shoulder_y)

    top_head = [
        int(shoulder_x),
        int(shoulder_y - torso_px * 0.50)
    ]

    heel = [
        int(shoulder_x),
        int(hip_y + torso_px * 1.625)
    ]

    front_measure = {

        "top_head": top_head,
        "heel": heel,

        "left_shoulder": sh_left,
        "right_shoulder": sh_right,

        "left_hip": hip_left,
        "right_hip": hip_right,

        "chest_left":
            front_kp.get(
                "chest_boundary_left"
            ),

        "chest_right":
            front_kp.get(
                "chest_boundary_right"
            ),

        "waist_left":
            front_kp.get(
                "waist_boundary_left"
            ),

        "waist_right":
            front_kp.get(
                "waist_boundary_right"
            )
    }

    # SIDE MAPPING

    chest_front = side_kp.get(
        "chest_boundary_left"
    )

    chest_back = side_kp.get(
        "chest_boundary_right"
    )

    waist_front = side_kp.get(
        "waist_boundary_left"
    )

    waist_back = side_kp.get(
        "waist_boundary_right"
    )

    hip_front = side_kp.get(
        "hip_boundary_left"
    )

    hip_back = side_kp.get(
        "hip_boundary_right"
    )

    if chest_front is None or chest_back is None:

        chest_front = side_kp.get(
            "shoulder_left"
        )

        chest_back = side_kp.get(
            "shoulder_right"
        )

    if hip_front is None or hip_back is None:

        hip_front = side_kp.get(
            "hip_left"
        )

        hip_back = side_kp.get(
            "hip_right"
        )

    side_measure = {

        "chest_front": chest_front,
        "chest_back": chest_back,

        "waist_front": waist_front,
        "waist_back": waist_back,

        "hip_front": hip_front,
        "hip_back": hip_back
    }

    # Measurements

    print("\nCalculating measurements...")

    results = calculate_measurements(

        front_keypoints=front_measure,

        side_keypoints=side_measure,

        real_height_cm=height_cm
    )

    print("\nBODY MEASUREMENTS\n")

    for k, v in results.items():
        print(k, ":", v, "cm")

    # Visualization

    for img in [front_path, side_path]:

        base, ext = os.path.splitext(
            os.path.basename(img)
        )

        out = os.path.join(
            output_dir,
            f"{base}_visualized{ext}"
        )

        engine.visualize_keypoints(
            img,
            out,
            show=False
        )

        print(out)


def main():

    engine = PoseEngine()

    validator = HumanPoseValidator()

    try:

        height_cm = float(
            input(
                "Enter height cm: "
            )
        )

    except:

        height_cm = 165

    output_dir = os.path.join(

        os.path.dirname(
            os.path.abspath(__file__)
        ),

        "..",
        "data",
        "output"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    front_path = os.path.join(
        BASE_DIR,
        "data",
        "samples",
        "front.jpeg"
    )

    side_path = os.path.join(
        BASE_DIR,
        "data",
        "samples",
        "sideprofile1.jpeg"
    )

    process_image_pair(
        front_path,
        side_path,
        engine,
        validator,
        height_cm,
        output_dir
    )


if __name__ == "__main__":
    main()