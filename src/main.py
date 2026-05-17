from pose_engine import PoseEngine
from validator import HumanPoseValidator
from geometry_engineer import calculate_measurements  # ← ADDED
import json
import os
import sys

def main():
    engine = PoseEngine()
    validator = HumanPoseValidator()

    # Ask for height once
    height_cm = float(input("Enter your height in cm: "))  # ← ADDED

    if len(sys.argv) > 1:
        images_to_process = sys.argv[1:]
    else:
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'samples')
        if os.path.exists(sample_dir):
            images_to_process = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
        else:
            images_to_process = ["fullbody.jpg", "test1.jpg"]

    any_processed = False
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'output')
    if not os.path.exists(output_dir) and '..' in output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for img_path in images_to_process:
        if os.path.exists(img_path):
            img_name = os.path.basename(img_path)
            print(f"\nProcessing {img_name}...")

            print("  [>] Running validation...")
            val_report = validator.validate_pose(img_path, expected_view="auto")

            if not val_report.get("valid", False):
                print(f"  [!] Validation failed for {img_name}:")
                for err in val_report.get("errors", []):
                    print(f"      - {err}")
                print("  [!] Skipping keypoint generation due to validation failure.")
                continue

            print(f"  [+] Validation passed ({val_report['view']} view detected).")

            keypoints = engine.get_keypoints(img_path)

            if isinstance(keypoints, str):
                print(f"Error in {img_name}: {keypoints}")
            else:
                print(f"Keypoints for {img_name} extracted successfully:")
                print(json.dumps(keypoints, indent=4))

                # ── GEOMETRY BLOCK START ─────────────────────────────
                kp = keypoints

                front_kp = {
                    'shoulder_left'  : kp.get('left_shoulder'),
                    'shoulder_right' : kp.get('right_shoulder'),
                    'hip_left'       : kp.get('left_hip'),
                    'hip_right'      : kp.get('right_hip'),
                }
                side_kp = front_kp.copy()

                measurements = calculate_measurements(front_kp, side_kp, height_cm)

                if measurements == 'invalid input':
                    print(f"  [!] Could not calculate measurements for {img_name}")
                else:
                    print(f"\n  MEASUREMENTS for {img_name}:")
                    print(f"    Chest : {measurements['chest']} cm")
                    print(f"    Waist : {measurements['waist']} cm")
                    print(f"    Hips  : {measurements['hips']}  cm")
                # ── GEOMETRY BLOCK END ───────────────────────────────

                base, ext = os.path.splitext(img_name)
                vis_name = f"{base}_visualized{ext}"
                vis_path = os.path.join(output_dir, vis_name) if os.path.exists(os.path.dirname(output_dir)) else vis_name

                engine.visualize_keypoints(img_path, vis_path, show=False)
                print(f"Visualization saved to '{vis_path}'")

            any_processed = True
        else:
            print(f"Error: File '{img_path}' not found.")

    if not any_processed:
        print("\n[!] No images processed.")
        print("Usage: python src/main.py <path_to_image>")

if __name__ == "__main__":
    main()