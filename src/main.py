from pose_engine import PoseEngine
from validator import HumanPoseValidator
from measurement_engineer import calculate_measurements
import json
import os
import sys
import math
# pyrefly: ignore [missing-import]
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def process_image_pair(front_path, side_path, engine, validator, height_cm, output_dir):

    # ── Validate both images ─────────────────────────────────────
    for img_path, view in [(front_path, "front"), (side_path, "side")]:
        if not os.path.exists(img_path):
            print(f"  [!] File not found: {img_path}")
            return
        print(f"\n[>] Validating {view} image: {os.path.basename(img_path)}")
        try:
            val = validator.validate_pose(img_path, expected_view=view)
        except Exception as e:
            print(f"  [!] Validation error: {e}")
            return
        if not val.get("valid", False):
            print(f"  [!] Validation failed ({view}):")
            for err in val.get("errors", []):
                print(f"      - {err}")
            print("  [i] Proceeding anyway for testing purposes...")
            # return
        print(f"  [+] {view} image OK")

    # ── Extract keypoints from both images ───────────────────────
    print("\n[>] Extracting keypoints...")
    try:
        front_kp = engine.get_keypoints(front_path)
        side_kp  = engine.get_keypoints(side_path)
    except Exception as e:
        print(f"  [!] Keypoint error: {e}")
        return

    if isinstance(front_kp, str):
        print(f"  [!] Front image error: {front_kp}")
        return
    if isinstance(side_kp, str):
        print(f"  [!] Side image error: {side_kp}")
        return

    print("  [+] Front keypoints:")
    print(json.dumps(front_kp, indent=4))
    print("  [+] Side keypoints:")
    print(json.dumps(side_kp, indent=4))

    # ── Build keypoint dicts for measurement_engineer ────────────
    # pose_engine gives us boundary points — map them to what
    # measurement_engineer.calculate_measurements() expects

    # HEAD: estimate from shoulder position (head is ~12% of height above shoulder)
    sh_left  = front_kp.get('shoulder_left')
    sh_right = front_kp.get('shoulder_right')
    hip_left  = front_kp.get('hip_left')
    hip_right = front_kp.get('hip_right')

    if not all([sh_left, sh_right, hip_left, hip_right]):
        print("  [!] Missing shoulder or hip keypoints from front image")
        return

    shoulder_y   = (sh_left[1]  + sh_right[1])  / 2
    shoulder_x   = (sh_left[0]  + sh_right[0])  / 2
    hip_y        = (hip_left[1] + hip_right[1]) / 2
    torso_px     = abs(hip_y - shoulder_y)

    # Estimate top of head (torso is ~32% of full height, head adds ~13%)
    top_head = [int(shoulder_x), int(shoulder_y - torso_px * 0.40)]

    # Estimate heel (legs are ~47% of full height below hip)
    heel     = [int(shoulder_x), int(hip_y + torso_px * 1.47)]

    # Front keypoints for measurement_engineer
    front_for_measurement = {
        "top_head"       : top_head,
        "heel"           : heel,
        "left_shoulder"  : sh_left,
        "right_shoulder" : sh_right,
        "left_hip"       : hip_left,
        "right_hip"      : hip_right,
    }

    # Side keypoints — use chest and hip boundary points from side image
    # pose_engine boundary points on side image = front-to-back depth
    chest_bl = side_kp.get('chest_boundary_left')
    chest_br = side_kp.get('chest_boundary_right')
    hip_bl   = side_kp.get('hip_boundary_left')
    hip_br   = side_kp.get('hip_boundary_right')

    # Fallback: if boundaries not detected, estimate from shoulder/hip joints
    if chest_bl is None or chest_br is None:
        print("  [!] Chest boundary not found in side image — using shoulder estimate")
        side_sh_l = side_kp.get('shoulder_left')
        side_sh_r = side_kp.get('shoulder_right')
        chest_bl  = side_sh_l
        chest_br  = side_sh_r

    if hip_bl is None or hip_br is None:
        print("  [!] Hip boundary not found in side image — using hip joint estimate")
        hip_bl = side_kp.get('hip_left')
        hip_br = side_kp.get('hip_right')

    if not all([chest_bl, chest_br, hip_bl, hip_br]):
        print("  [!] Missing side image boundary points — cannot calculate depth")
        return

    side_for_measurement = {
        "chest_front" : chest_bl,
        "chest_back"  : chest_br,
        "hip_front"   : hip_bl,
        "hip_back"    : hip_br,
    }

    # ── Run measurements ─────────────────────────────────────────
    print("\n[>] Calculating measurements...")
    try:
        results = calculate_measurements(
            front_keypoints = front_for_measurement,
            side_keypoints  = side_for_measurement,
            real_height_cm  = height_cm
        )
        print(f"\n  ╔══════════════════════════════════╗")
        print(f"  ║       BODY MEASUREMENTS          ║")
        print(f"  ╠══════════════════════════════════╣")
        for key, val in results.items():
            print(f"  ║  {key:<18} : {val} cm")
        print(f"  ╚══════════════════════════════════╝")
    except Exception as e:
        print(f"  [!] Measurement error: {e}")
        return

    # ── Visualize both images ────────────────────────────────────
    for img_path in [front_path, side_path]:
        try:
            base, ext = os.path.splitext(os.path.basename(img_path))
            vis_path  = os.path.join(output_dir, f"{base}_visualized{ext}")
            engine.visualize_keypoints(img_path, vis_path, show=False)
            print(f"  [+] Saved: {vis_path}")
        except Exception as e:
            print(f"  [!] Visualization error: {e}")


def main():
    engine    = PoseEngine()
    validator = HumanPoseValidator()

    try:
        height_cm = float(input("Enter your height in cm: "))
    except ValueError:
        print("[!] Invalid. Using 165 cm.")
        height_cm = 165.0

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'output'
    )
    os.makedirs(output_dir, exist_ok=True)

    # ── How to provide images ────────────────────────────────────
    if len(sys.argv) == 3:
        # Pass two images directly:
        # python src/main.py front.jpg side.jpg
        front_path = sys.argv[1]
        side_path  = sys.argv[2]

    elif len(sys.argv) == 2:
        # Only one image passed — ask for the other
        front_path = sys.argv[1]
        side_path  = input("Enter side image path: ").strip()

    else:
        # No arguments — ask for both
        print("\nNo image paths provided.")
        print("Tip: run as  python src/main.py <front_img> <side_img>")
        front_path = input("Enter FRONT image path: ").strip()
        side_path  = input("Enter SIDE  image path: ").strip()

        # If still empty, fall back to samples folder
        if not front_path:
            sample_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'samples'
            )
            samples = [
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ] if os.path.exists(sample_dir) else []

            if len(samples) >= 2:
                front_path = samples[0]
                side_path  = samples[1]
                print(f"  Using: {front_path}")
                print(f"  Using: {side_path}")
            else:
                print("[!] Not enough sample images found.")
                return

    process_image_pair(
        front_path.strip(),
        side_path.strip(),
        engine, validator, height_cm, output_dir
    )

    print("\n[✓] Done.")

if __name__ == "__main__":
    main()