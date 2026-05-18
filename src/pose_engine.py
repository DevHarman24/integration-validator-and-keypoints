import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class PoseEngine:
    def __init__(self):
        # Initialize MediaPipe Tasks Pose Landmarker
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'models', 'pose_landmarker.task')
        
        if not os.path.exists(model_path):
            # Fallback for root execution
            model_path = os.path.join(current_dir, 'models', 'pose_landmarker.task')
            if not os.path.exists(model_path):
                model_path = 'pose_landmarker.task'
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def get_person_mask_grabcut(self, image, landmarks, h, w, is_side=False, mp_mask=None):
        """
        Generates a person mask using a two-pass GrabCut approach, optionally initialized with MediaPipe mask.
        """
        mask = np.zeros(image.shape[:2], np.uint8)
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        rect = None
        if mp_mask is not None:
            # Initialize with MediaPipe's highly accurate mask
            mask[mp_mask > 0.5] = cv2.GC_PR_FGD
            mask[mp_mask <= 0.5] = cv2.GC_BGD
        else:
            # 1. Define Bounding Box for the person
            xs = [int(l.x * w) for l in landmarks]
            ys = [int(l.y * h) for l in landmarks]
            rect = (max(0, min(xs) - 60), max(0, min(ys) - 60), 
                    min(w-1, max(xs) + 60) - max(0, min(xs) - 60), 
                    min(h-1, max(ys) + 40) - max(0, min(ys) - 60))
            
            # Pass 1: Initialize with Rectangle
            try:
                cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            except:
                pass
            
        # Pass 2: Refine with Skeleton Guidance and Arm Masking
        # Torso Polygon (Definite Foreground)
        sh_l = [int(landmarks[11].x * w), int(landmarks[11].y * h)]
        sh_r = [int(landmarks[12].x * w), int(landmarks[12].y * h)]
        hip_l = [int(landmarks[23].x * w), int(landmarks[23].y * h)]
        hip_r = [int(landmarks[24].x * w), int(landmarks[24].y * h)]
        torso_pts = np.array([sh_l, sh_r, hip_r, hip_l], np.int32)
        cv2.fillPoly(mask, [torso_pts], cv2.GC_FGD)
        
        # Spine (Definite Foreground)
        spine_top = ((sh_l[0]+sh_r[0])//2, (sh_l[1]+sh_r[1])//2)
        spine_bottom = ((hip_l[0]+hip_r[0])//2, (hip_l[1]+hip_r[1])//2)
        cv2.line(mask, spine_top, spine_bottom, cv2.GC_FGD, 10)
        
        # Arms (Definite Background) to ensure they are separated from torso
        # Only do this for front view. For side view, arms overlap torso and we need full depth.
        if not is_side:
            arm_landmarks = [[11, 13, 15], [12, 14, 16]]
            for indices in arm_landmarks:
                pts = [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices]
                for i in range(len(pts)-1):
                    cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), cv2.GC_BGD, 40)

        try:
            mode = cv2.GC_INIT_WITH_MASK
            cv2.grabCut(image, mask, None, bgdModel, fgdModel, 3, mode)
        except:
            pass
            
        person_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return person_mask

    def get_visual_boundary(self, mask, y, center_x):
        """
        Finds the left and right edges of the person at height y, 
        selecting the island that contains center_x.
        """
        row = mask[y]
        diff = np.diff(row.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        if row[0] == 1: starts = np.insert(starts, 0, 0)
        if row[-1] == 1: ends = np.append(ends, len(row))
        
        if len(starts) == 0:
            return None, None
            
        # Find the island that contains center_x
        for s, e in zip(starts, ends):
            if s <= center_x <= e:
                return int(s), int(e)
        
        # If no island contains center_x, pick the closest one
        closest_idx = np.argmin([min(abs(s - center_x), abs(e - center_x)) for s, e in zip(starts, ends)])
        return int(starts[closest_idx]), int(ends[closest_idx])

    def get_keypoints(self, image_path, skip_boundaries=False):
        """
        Processes an image and returns key body landmarks and visual boundaries.
        Set skip_boundaries=True to bypass GrabCut for faster skeleton evaluation.
        """
        if not os.path.exists(image_path):
            return "Error: File not found."

        image_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_raw is None:
            return "Error: Could not load image."
        
        orig_h, orig_w = image_raw.shape[:2]
        
        # 1. Internal Downscaling for performance (max 1280px)
        max_dim = 1280
        scale = 1.0
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / max(orig_h, orig_w)
            image_cv = cv2.resize(image_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            image_cv = image_raw.copy()

        h, w = image_cv.shape[:2]
        
        if len(image_cv.shape) == 2:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)
        elif image_cv.shape[2] == 4:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)
        
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.pose_landmarks:
            return "Error: No pose detected in the image."

        landmarks = detection_result.pose_landmarks[0]

        # Internal joints for calculation
        temp_joints = {
            "shoulder_left": [int(landmarks[11].x * w), int(landmarks[11].y * h)],
            "shoulder_right": [int(landmarks[12].x * w), int(landmarks[12].y * h)],
            "elbow_left": [int(landmarks[13].x * w), int(landmarks[13].y * h)],
            "elbow_right": [int(landmarks[14].x * w), int(landmarks[14].y * h)],
            "knee_left": [int(landmarks[25].x * w), int(landmarks[25].y * h)],
            "knee_right": [int(landmarks[26].x * w), int(landmarks[26].y * h)],
            "hip_left": [int(landmarks[23].x * w), int(landmarks[23].y * h)],
            "hip_right": [int(landmarks[24].x * w), int(landmarks[24].y * h)]
        }

        # Detect if it's a side view or front view
        # Robust view detection using Shoulder-Width to Torso-Height ratio
        # In front view, shoulders are ~40-50% of torso height. In side view, they are very narrow.
        shoulder_dist_norm = abs(landmarks[11].x - landmarks[12].x)
        torso_height_norm = abs(landmarks[11].y - landmarks[23].y)
        is_side = (shoulder_dist_norm / max(torso_height_norm, 0.01)) < 0.35
        center_x = int((temp_joints["hip_left"][0] + temp_joints["hip_right"][0] + 
                        temp_joints["shoulder_left"][0] + temp_joints["shoulder_right"][0]) / 4)

        mp_mask = None
        if detection_result.segmentation_masks:
            mp_mask = detection_result.segmentation_masks[0].numpy_view()
            if len(mp_mask.shape) > 2:
                mp_mask = np.squeeze(mp_mask)

        # Generate mask using GrabCut (unless skipped)
        person_mask = None
        if not skip_boundaries:
            if is_side and mp_mask is not None:
                # For side view, use the highly accurate MediaPipe mask directly to prevent background bleeding
                person_mask = (mp_mask > 0.5).astype(np.uint8)
            else:
                person_mask = self.get_person_mask_grabcut(image_cv, landmarks, h, w, is_side, mp_mask)

        # Result dictionary (using downscaled coords temporarily)
        res_downscaled = {
            "shoulder_left": temp_joints["shoulder_left"],
            "shoulder_right": temp_joints["shoulder_right"],
            "hip_left": temp_joints["hip_left"],
            "hip_right": temp_joints["hip_right"],
            "knee_left": temp_joints["knee_left"],
            "knee_right": temp_joints["knee_right"]
        }

        # Calculate heights for reference based on model skeleton
        shoulder_y = int((temp_joints["shoulder_left"][1] + temp_joints["shoulder_right"][1]) / 2)
        hip_y = int((temp_joints["hip_left"][1] + temp_joints["hip_right"][1]) / 2)

        # Initialize boundaries as None
        best_chest_l = best_chest_r = None
        best_waist_l = best_waist_r = None
        best_hip_l = best_hip_r = None

        if not skip_boundaries and person_mask is not None:
            if is_side:
                # --- SIDE VIEW LOGIC (Calculates Body Depth) ---
                # For side poses, the distance between the boundaries represents the depth of the body.
                
                # 1. Exact Chest Depth (~25% down the torso from shoulders)
                chest_y = int(shoulder_y + 0.25 * (hip_y - shoulder_y))
                c_l, c_r = self.get_visual_boundary(person_mask, chest_y, center_x)
                best_chest_l = [c_l, chest_y] if c_l is not None else None
                best_chest_r = [c_r, chest_y] if c_r is not None else None
    
                # 2. Exact Waist Depth (~58% down the torso from shoulders, pose-agnostic)
                waist_y = int(shoulder_y + 0.58 * (hip_y - shoulder_y))
                w_l, w_r = self.get_visual_boundary(person_mask, waist_y, center_x)
                best_waist_l = [w_l, waist_y] if w_l is not None else None
                best_waist_r = [w_r, waist_y] if w_r is not None else None
    
                # 3. Exact Hip Depth (at the hip joint level)
                h_l, h_r = self.get_visual_boundary(person_mask, hip_y, center_x)
                best_hip_l = [h_l, hip_y] if h_l is not None else None
                best_hip_r = [h_r, hip_y] if h_r is not None else None
            else:
                # --- FRONT VIEW LOGIC (Calculates Body Width) ---
                
                # 1. Exact Chest Calculation (~25% down the torso from shoulders)
                chest_y = int(shoulder_y + 0.25 * (hip_y - shoulder_y))
                c_l, c_r = self.get_visual_boundary(person_mask, chest_y, center_x)
                best_chest_l = [c_l, chest_y] if c_l is not None else None
                best_chest_r = [c_r, chest_y] if c_r is not None else None
    
                # 2. Exact Waist Calculation (~60% down the torso from shoulders, pose-agnostic)
                waist_y = int(shoulder_y + 0.60 * (hip_y - shoulder_y))
                w_l, w_r = self.get_visual_boundary(person_mask, waist_y, center_x)
                best_waist_l = [w_l, waist_y] if w_l is not None else None
                best_waist_r = [w_r, waist_y] if w_r is not None else None
    
                # 3. Exact Hip Calculation (at the hip joint level)
                h_l, h_r = self.get_visual_boundary(person_mask, hip_y, center_x)
                best_hip_l = [h_l, hip_y] if h_l is not None else None
                best_hip_r = [h_r, hip_y] if h_r is not None else None

        # Add calculated boundaries
        res_downscaled["chest_boundary_left"] = best_chest_l
        res_downscaled["chest_boundary_right"] = best_chest_r
        res_downscaled["waist_boundary_left"] = best_waist_l
        res_downscaled["waist_boundary_right"] = best_waist_r
        res_downscaled["hip_boundary_left"] = best_hip_l
        res_downscaled["hip_boundary_right"] = best_hip_r

        res_downscaled["view_type"] = "side" if is_side else "front"



        # Scale everything back UP to original coordinates
        keypoints_dict = {}
        for name, coord in res_downscaled.items():
            if name == "view_type":
                keypoints_dict[name] = coord
            elif coord is not None:
                keypoints_dict[name] = [int(coord[0] / scale), int(coord[1] / scale)]
            else:
                keypoints_dict[name] = None

        return keypoints_dict

        return keypoints_dict

    def visualize_keypoints(self, image_path, output_path, show=False):
        """
        Draws keypoints on the image for verification and optionally shows it.
        """
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Could not load image."

        keypoints = self.get_keypoints(image_path)
        if isinstance(keypoints, str):
            return keypoints

        for name, coord in keypoints.items():
            # Only draw if it's a coordinate pair (list of 2 numbers)
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                cv2.circle(image, (coord[0], coord[1]), 8, (0, 255, 0), -1)
                cv2.putText(image, name, (coord[0] + 12, coord[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(image, name, (coord[0] + 12, coord[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Optional Pop-up Preview
        if show:
            window_name = f"Keypoint Verification - {os.path.basename(image_path)}"
            # Use WINDOW_NORMAL to fit on screen, but KEEPRATIO to prevent stretching
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(window_name, image)
            
            print(f"\n[!] IMAGE READY: Please CLICK ON THE WINDOW '{window_name}' first, then press ANY KEY to close it and continue.")
            
            # Wait for any key or for the window to be closed
            while True:
                # Check if window is still open
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                # Wait 100ms for a key press
                if cv2.waitKey(100) & 0xFF != 255:
                    break
            
            cv2.destroyAllWindows()

        cv2.imwrite(output_path, image)

        return f"Visualization saved to {output_path}"
