import cv2
import mediapipe as mp
import os
import sys
import numpy as np
import json

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HumanPoseValidator:
    def __init__(self):
        print(f"DEBUG: MediaPipe version: {mp.__version__}")
        
        # Initialize MediaPipe Tasks Pose Landmarker
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'models', 'pose_landmarker.task')
        
        if not os.path.exists(model_path):
            model_path = os.path.join(current_dir, 'models', 'pose_landmarker.task')
            if not os.path.exists(model_path):
                model_path = 'pose_landmarker.task'
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def check_quality(self, image):
        """Checks for blur and lighting issues."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur detection using Laplacian variance
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = blur_value < 40  # Lowered from 100 to be more lenient
        
        # Lighting detection
        avg_brightness = np.mean(gray)
        is_dark = avg_brightness < 60  # Require good lighting
        is_bright = avg_brightness > 230  # Prevent overexposure
        
        return {
            "blur_score": blur_value,
            "is_blurry": bool(is_blurry),
            "brightness": avg_brightness,
            "is_dark": bool(is_dark),
            "is_bright": bool(is_bright)
        }

    def validate_pose(self, image_path, expected_view="front"):
        """Validates if the image matches the expected pose (front or side)."""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not read image at {image_path}"}

        # 1. Quality Check
        quality = self.check_quality(image)
        
        # 2. Pose Detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)

        if not results.pose_landmarks:
            return {
                "valid": False,
                "quality": quality,
                "errors": ["No human detected in the image."]
            }

        landmarks = results.pose_landmarks[0]
        errors = []
        
        # Helper to get visibility and coordinates
        def get_lm(id):
            return landmarks[id]

        if expected_view == "auto":
            # Auto-detect view based on Shoulder-Width to Torso-Height ratio
            l_sh_auto, r_sh_auto = get_lm(11), get_lm(12)
            l_hip_auto = get_lm(23)
            shoulder_dist_norm = abs(l_sh_auto.x - r_sh_auto.x)
            torso_height_norm = abs(l_sh_auto.y - l_hip_auto.y)
            is_side = (shoulder_dist_norm / max(torso_height_norm, 0.01)) < 0.35
            expected_view = "side" if is_side else "front"

        # 3. Check Visibility
        # For Front view, we want full body. For Side view, we are more lenient with ankles.
        if expected_view == "front":
            required_ids = [0, 11, 12, 23, 24, 27, 28] # Full body
        else:
            required_ids = [0, 11, 12, 23, 24] # Head, Shoulders, Hips (Ankles optional)
            
        visibility_threshold = 0.05
        
        missing_parts = False
        for i in required_ids:
            if landmarks[i].visibility < visibility_threshold:
                missing_parts = True
                break
        
        if missing_parts:
            errors.append("Full body is not visible. Ensure head, shoulders, hips, and ankles are in frame.")

        # 4. Extract Key Landmarks for Checks
        nose = get_lm(0)
        l_sh, r_sh = get_lm(11), get_lm(12)
        l_hip, r_hip = get_lm(23), get_lm(24)
        l_wrist, r_wrist = get_lm(15), get_lm(16)
        l_ear, r_ear = get_lm(7), get_lm(8)
        
        # 5. Check for Tilted/Complex Poses (Strict Verticality)
        # Shoulder Levelness
        shoulder_tilt = abs(l_sh.y - r_sh.y)
        if shoulder_tilt > 0.12: 
            errors.append("Body is tilted. Please keep your shoulders level.")
            
        # Hip Levelness
        hip_tilt = abs(l_hip.y - r_hip.y)
        if hip_tilt > 0.08:
            errors.append("Hips are tilted. Please stand straight.")

        # Arm Position Check
        # HARD RULE: Hands must never be above the head/nose
        if l_wrist.y < nose.y or r_wrist.y < nose.y:
            errors.append("Hands are raised too high. Please keep arms at your sides.")
            
        # Standard arm position check (must be below shoulders in Front View)
        if expected_view == "front":
            if l_wrist.y < l_sh.y or r_wrist.y < r_sh.y:
                errors.append("Arms must be at your sides. Do not raise your hands.")

        # 6. View Specific Checks
        shoulder_width = abs(l_sh.x - r_sh.x)
        
        if expected_view == "front":
            mid_shoulder_x = (l_sh.x + r_sh.x) / 2
            mid_hip_x = (l_hip.x + r_hip.x) / 2
            
            # Spine Alignment: Head should be aligned with center of hips
            spine_tilt = abs(nose.x - mid_hip_x)
            if spine_tilt > 0.25: # Strict for front view
                errors.append("Body/Spine is tilted. Please stand up straight.")

            nose_offset = abs(nose.x - mid_shoulder_x)
            if nose_offset > 0.2: 
                errors.append("Body is off-center or tilted.")
            
            if shoulder_width < 0.12: # Tightened from 0.08
                errors.append("Shoulders appear too narrow.")
                
            # Leg Crossing Check (Ankles should not be too close together)
            l_ank, r_ank = get_lm(27), get_lm(28)
            ankle_dist = abs(l_ank.x - r_ank.x)
            if ankle_dist < 0.05: # Legs are crossed or too close
                errors.append("Legs must be straight and slightly apart. Do not cross your legs.")

            # Knee Alignment Check (Should be horizontally level)
            l_knee, r_knee = get_lm(25), get_lm(26)
            knee_tilt = abs(l_knee.y - r_knee.y)
            if knee_tilt > 0.08:
                errors.append("Knees are not level. Please stand evenly on both legs.")

            if l_ear.visibility < 0.3 or r_ear.visibility < 0.3:
                errors.append("Face is not looking straight ahead.")

        elif expected_view == "side":
            # 7. Anatomical "Plumb Line" Check (Ear, Shoulder, Hip, Knee, Ankle Alignment)
            ear, shoulder = get_lm(7), get_lm(11)
            hip, knee, ankle = get_lm(23), get_lm(25), get_lm(27)
            
            # Use ear as the vertical anchor
            # In a true profile, these should all have very similar X coordinates
            plumb_errors = []
            if abs(ear.x - shoulder.x) > 0.15: plumb_errors.append("Ear-Shoulder")
            if abs(shoulder.x - hip.x) > 0.15: plumb_errors.append("Shoulder-Hip")
            if abs(hip.x - knee.x) > 0.15: plumb_errors.append("Hip-Knee")
            if abs(knee.x - ankle.x) > 0.15: plumb_errors.append("Knee-Ankle")
            
            if plumb_errors:
                errors.append(f"Anatomical Plumb Line deviation: {', '.join(plumb_errors)}. Please stand straight.")
                
            # Torso Depth Check (ensures it's a profile)
            if shoulder_width > 0.3:
                errors.append("Torso appears too wide. Please turn 90 degrees for a true side profile.")

        # 5. Quality Errors
        if quality["is_blurry"]:
            errors.append("Image is too blurry.")
        if quality["is_dark"]:
            errors.append("Lighting is too dark.")
        if quality["is_bright"]:
            errors.append("Lighting is too bright (overexposed).")

        return {
            "valid": len(errors) == 0,
            "quality": quality,
            "view": expected_view,
            "errors": errors,
            "landmarks": landmarks # Kept for internal processing, removed before final output
        }

    def _get_body_proportions(self, landmarks):
        """Calculates ratios of body parts to check for consistency."""
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        l_ank = landmarks[27]
        r_ank = landmarks[28]

        # Torso height (average of left/right)
        torso = ((l_hip.y - l_sh.y) + (r_hip.y - r_sh.y)) / 2
        # Leg length (average of left/right)
        legs = ((l_ank.y - l_hip.y) + (r_ank.y - r_hip.y)) / 2
        
        if legs == 0: return 0
        return torso / legs

    def process_pair(self, front_image_path, side_image_path):
        """Processes both images and returns a combined validation report."""
        front_report = self.validate_pose(front_image_path, "front")
        side_report = self.validate_pose(side_image_path, "side")
        
        overall_errors = []
        same_person = True

        if front_report.get("valid") and side_report.get("valid"):
            # Same Person Heuristic: Strict proportion check
            f_prop = self._get_body_proportions(front_report["landmarks"])
            s_prop = self._get_body_proportions(side_report["landmarks"])
            
            prop_diff = abs(f_prop - s_prop)
            if prop_diff > 0.12: # Strict 12% tolerance
                same_person = False
                overall_errors.append("Same Person Check: Body proportions do not match perfectly.")

        # Cleanup landmarks from reports for JSON serializability
        if "landmarks" in front_report: del front_report["landmarks"]
        if "landmarks" in side_report: del side_report["landmarks"]

        if not front_report.get("valid", False):
            overall_errors.extend([f"Front Image: {e}" for e in front_report.get("errors", [])])
        if not side_report.get("valid", False):
            overall_errors.extend([f"Side Image: {e}" for e in side_report.get("errors", [])])

        result = {
            "status": "success" if not overall_errors else "failed",
            "front_pose_valid": front_report.get("valid", False),
            "side_pose_valid": side_report.get("valid", False),
            "same_person_check": same_person,
            "errors": overall_errors,
            "details": {
                "front": front_report,
                "side": side_report
            }
        }
        return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("\n[!] Usage: python validator.py <front_image_path> <side_image_path>")
        sys.exit(1)
    
    front_path = sys.argv[1]
    side_path = sys.argv[2]
    
    validator = HumanPoseValidator()
    if os.path.exists(front_path) and os.path.exists(side_path):
        report = validator.process_pair(front_path, side_path)
        
        # --- HUMAN READABLE MESSAGE ---
        print("\n" + "="*50)
        print("         3D AI MODEL - IMAGE VALIDATION")
        print("="*50)
        
        # Front View Status
        status_front = "[PASS]" if report["front_pose_valid"] else "[FAIL]"
        print(f"\nFRONT VIEW: {status_front}")
        if not report["front_pose_valid"]:
            for err in report["details"]["front"]["errors"]:
                print(f"  - {err}")
        
        # Side View Status
        status_side = "[PASS]" if report["side_pose_valid"] else "[FAIL]"
        print(f"\nSIDE VIEW:  {status_side}")
        if not report["side_pose_valid"]:
            for err in report["details"]["side"]["errors"]:
                print(f"  - {err}")

        # Same Person Check
        status_person = "[VERIFIED]" if report["same_person_check"] else "[WARNING]"
        print(f"\nSAME PERSON CHECK: {status_person}")
        if not report["same_person_check"]:
            print("  - The body proportions between images do not match.")

        print("\n" + "="*50)
        if report["status"] == "success":
            print("   RESULT: Images are READY for 3D Reconstruction!")
        else:
            print("   RESULT: Please fix the errors above and try again.")
        print("="*50 + "\n")
        
    else:
        print("\n[!] Error: One or both image files do not exist.")
        print(f"    Paths checked:\n    1. {front_path}\n    2. {side_path}")
