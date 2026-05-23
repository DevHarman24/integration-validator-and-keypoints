import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os


class PoseEngine:

    def __init__(self):

        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        model_path = os.path.join(

            current_dir,

            "..",

            "models",

            "pose_landmarker.task"
        )

        if not os.path.exists(model_path):

            model_path = os.path.join(

                current_dir,

                "models",

                "pose_landmarker.task"
            )

            if not os.path.exists(model_path):

                model_path = (
                    "pose_landmarker.task"
                )

        base_options = python.BaseOptions(

            model_asset_path=model_path
        )

        options = vision.PoseLandmarkerOptions(

            base_options=base_options,

            output_segmentation_masks=True
        )

        self.detector = (
            vision
            .PoseLandmarker
            .create_from_options(
                options
            )
        )

    def get_person_mask_grabcut(

        self,

        image,

        landmarks,

        h,

        w,

        is_side=False,

        mp_mask=None
    ):

        mask = np.zeros(

            image.shape[:2],

            np.uint8
        )

        bgdModel = np.zeros(
            (1,65),
            np.float64
        )

        fgdModel = np.zeros(
            (1,65),
            np.float64
        )

        if mp_mask is not None:

            mask[
                mp_mask > 0.5
            ] = cv2.GC_PR_FGD

            mask[
                mp_mask <= 0.5
            ] = cv2.GC_BGD

        else:

            xs = [
                int(l.x*w)
                for l in landmarks
            ]

            ys = [
                int(l.y*h)
                for l in landmarks
            ]

            rect = (

                max(
                    0,
                    min(xs)-60
                ),

                max(
                    0,
                    min(ys)-60
                ),

                min(
                    w-1,
                    max(xs)+60
                ) -

                max(
                    0,
                    min(xs)-60
                ),

                min(
                    h-1,
                    max(ys)+40
                ) -

                max(
                    0,
                    min(ys)-60
                )
            )

            try:

                cv2.grabCut(

                    image,

                    mask,

                    rect,

                    bgdModel,

                    fgdModel,

                    3,

                    cv2.GC_INIT_WITH_RECT
                )

            except:

                pass

        person_mask = np.where(

            (
                mask==2
            )

            |

            (
                mask==0
            ),

            0,

            1

        ).astype(
            "uint8"
        )

        return person_mask

    def get_visual_boundary(

        self,

        mask,

        y,

        center_x
    ):

        row = mask[y]

        diff = np.diff(
            row.astype(
                np.int8
            )
        )

        starts = (
            np.where(
                diff==1
            )[0]+1
        )

        ends = (
            np.where(
                diff==-1
            )[0]+1
        )

        if len(starts)==0:

            return None,None

        for s,e in zip(
            starts,
            ends
        ):

            if s<=center_x<=e:

                return int(s),int(e)

        return None,None

    def get_keypoints(

        self,

        image_path,

        skip_boundaries=False
    ):

        if not os.path.exists(
            image_path
        ):

            return (
                "Error: File not found."
            )

        image_raw = cv2.imread(

            image_path,

            cv2.IMREAD_UNCHANGED
        )

        if image_raw is None:

            return (
                "Error: Could not load image."
            )

        # grayscale fix

        if len(
            image_raw.shape
        )==2:

            image_raw = (
                cv2.cvtColor(

                    image_raw,

                    cv2.COLOR_GRAY2BGR
                )
            )

        # RGBA fix

        elif (

            len(
                image_raw.shape
            )==3

            and

            image_raw.shape[2]==4
        ):

            image_raw = (
                cv2.cvtColor(

                    image_raw,

                    cv2.COLOR_BGRA2BGR
                )
            )

        image_raw = image_raw.astype(
            np.uint8
        )

        orig_h,orig_w = (
            image_raw.shape[:2]
        )

        image_cv = image_raw.copy()

        h,w = image_cv.shape[:2]

        image_rgb = cv2.cvtColor(

            image_cv,

            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(

            image_format=
            mp.ImageFormat.SRGB,

            data=image_rgb
        )

        try:

            detection_result = (
                self.detector.detect(
                    mp_image
                )
            )

        except Exception as e:

            print(e)

            return "POSE_FAIL"

        if not detection_result.pose_landmarks:

            return (
                "Error: No pose detected."
            )

        landmarks = (
            detection_result
            .pose_landmarks[0]
        )

        temp_joints = {

            "shoulder_left":[

                int(
                    landmarks[11].x*w
                ),

                int(
                    landmarks[11].y*h
                )
            ],

            "shoulder_right":[

                int(
                    landmarks[12].x*w
                ),

                int(
                    landmarks[12].y*h
                )
            ],

            "hip_left":[

                int(
                    landmarks[23].x*w
                ),

                int(
                    landmarks[23].y*h
                )
            ],

            "hip_right":[

                int(
                    landmarks[24].x*w
                ),

                int(
                    landmarks[24].y*h
                )
            ],

            "knee_left":[

                int(
                    landmarks[25].x*w
                ),

                int(
                    landmarks[25].y*h
                )
            ],

            "knee_right":[

                int(
                    landmarks[26].x*w
                ),

                int(
                    landmarks[26].y*h
                )
            ]
        }

        return temp_joints

    def visualize_keypoints(

        self,

        image_path,

        output_path,

        show=False
    ):

        image = cv2.imread(
            image_path
        )

        if image is None:

            return (
                "Error loading image"
            )

        keypoints = (
            self.get_keypoints(
                image_path
            )
        )

        if isinstance(
            keypoints,
            str
        ):

            return keypoints

        for name,coord in keypoints.items():

            cv2.circle(

                image,

                tuple(coord),

                8,

                (0,255,0),

                -1
            )

            cv2.putText(

                image,

                name,

                (
                    coord[0]+10,

                    coord[1]
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255,255,255),

                1
            )

        cv2.imwrite(
            output_path,
            image
        )

        return (
            f"Saved {output_path}"
        )