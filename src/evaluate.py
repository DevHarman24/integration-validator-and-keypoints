"""
=============================================================================
  Pose Engine Accuracy & Precision Evaluator
  -------------------------------------------
  Evaluates the PoseEngine against COCO 2017 Val ground truth.

  Usage:
      python src/evaluate.py                  # Uses full dataset (up to 300 images)
      python src/evaluate.py --max-images 50  # Quick run with 50 images
      python src/evaluate.py --skip-download  # Skip download if files exist

  Output:
      data/eval_results/report.txt      -- Human-readable summary
      data/eval_results/per_joint.csv   -- Per-image, per-joint error table
=============================================================================
"""

import os
import sys
import json
import csv
import math
import argparse
import zipfile
import time

# Patch sys.path so we can import PoseEngine from the same src/ folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
EVAL_DIR     = os.path.join(DATA_DIR, "eval_results")
COCO_DIR     = os.path.join(DATA_DIR, "coco_eval")
IMAGES_DIR   = os.path.join(COCO_DIR, "val2017")
ANNO_PATH    = os.path.join(COCO_DIR, "annotations", "person_keypoints_val2017.json")

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOT_URL  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO joint index -> engine key mapping
#   5 = left_shoulder   6 = right_shoulder
#   11 = left_hip       12 = right_hip
#   13 = left_knee      14 = right_knee
JOINT_MAP = {
    "shoulder_left" : 5,
    "shoulder_right": 6,
    "hip_left"      : 11,
    "hip_right"     : 12,
    "knee_left"     : 13,
    "knee_right"    : 14,
}

PCK_THRESHOLDS = [10, 20, 50]


# =============================================================================
#   DOWNLOAD HELPERS
# =============================================================================

def download_file(url, dest):
    try:
        import requests
    except ImportError:
        print("[!] 'requests' not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests

    print("\n  Downloading: " + os.path.basename(dest))
    print("  From       : " + url)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB chunks

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    done = int(pct / 2)
                    bar = "#" * done + "-" * (50 - done)
                    sys.stdout.write("\r  [%s] %5.1f%%  %dMB/%dMB" % (
                        bar, pct, downloaded // 1024 // 1024, total // 1024 // 1024))
                    sys.stdout.flush()
    print("")


def extract_zip(zip_path, dest_dir):
    sys.stdout.write("  Extracting " + os.path.basename(zip_path) + " ... ")
    sys.stdout.flush()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("Done.")


def ensure_coco_data(skip_download=False):
    os.makedirs(COCO_DIR, exist_ok=True)

    images_zip = os.path.join(COCO_DIR, "val2017.zip")
    if not os.path.isdir(IMAGES_DIR):
        if skip_download:
            print("[!] Images folder not found and --skip-download is set. Aborting.")
            sys.exit(1)
        if not os.path.isfile(images_zip):
            download_file(IMAGES_URL, images_zip)
        extract_zip(images_zip, COCO_DIR)
    else:
        print("  [OK] COCO val2017 images already present.")

    anno_zip = os.path.join(COCO_DIR, "annotations_trainval2017.zip")
    if not os.path.isfile(ANNO_PATH):
        if skip_download:
            print("[!] Annotations not found and --skip-download is set. Aborting.")
            sys.exit(1)
        if not os.path.isfile(anno_zip):
            download_file(ANNOT_URL, anno_zip)
        extract_zip(anno_zip, COCO_DIR)
    else:
        print("  [OK] COCO annotations already present.")


# =============================================================================
#   COCO ANNOTATION LOADER  (pure JSON, no pycocotools needed)
# =============================================================================

def load_coco_annotations(max_images):
    sys.stdout.write("\n  Loading COCO annotations ... ")
    sys.stdout.flush()
    with open(ANNO_PATH, "r") as f:
        coco = json.load(f)
    print("Done.")

    img_map = {img["id"]: img for img in coco["images"]}

    from collections import defaultdict
    anno_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("num_keypoints", 0) > 0:
            anno_by_image[ann["image_id"]].append(ann)

    results = []
    for image_id, anns in anno_by_image.items():
        if len(anns) != 1:
            continue

        ann = anns[0]
        img_info = img_map.get(image_id)
        if img_info is None:
            continue

        bbox = ann.get("bbox", [0, 0, 0, 0])
        if bbox[3] < 100:
            continue

        kps_flat = ann["keypoints"]
        gt_joints = {}
        valid = True
        for joint_name, coco_idx in JOINT_MAP.items():
            x = kps_flat[coco_idx * 3]
            y = kps_flat[coco_idx * 3 + 1]
            v = kps_flat[coco_idx * 3 + 2]
            if v < 1:
                valid = False
                break
            gt_joints[joint_name] = [x, y]

        if not valid:
            continue

        results.append({
            "image_id": image_id,
            "file_name": img_info["file_name"],
            "width": img_info["width"],
            "height": img_info["height"],
            "gt_joints": gt_joints,
        })

        if len(results) >= max_images:
            break

    print("  Found %d usable images (single-person, all 6 joints visible)." % len(results))
    return results


# =============================================================================
#   METRIC CALCULATIONS
# =============================================================================

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_metrics(errors_by_joint):
    pck      = {t: {} for t in PCK_THRESHOLDS}
    mean_err = {}
    std_err  = {}

    for joint, errs in errors_by_joint.items():
        if not errs:
            continue
        n    = len(errs)
        mean = sum(errs) / n
        variance = sum((e - mean) ** 2 for e in errs) / n
        std  = math.sqrt(variance)

        mean_err[joint] = mean
        std_err[joint]  = std

        for t in PCK_THRESHOLDS:
            correct = sum(1 for e in errs if e <= t)
            pck[t][joint] = correct / n * 100

    for t in PCK_THRESHOLDS:
        vals = list(pck[t].values())
        pck[t]["overall"] = sum(vals) / len(vals) if vals else 0

    return pck, mean_err, std_err


# =============================================================================
#   REPORT GENERATION  (100% ASCII - safe on all Windows consoles)
# =============================================================================

def save_report(pck, mean_err, std_err, n_images, n_failed, elapsed):
    os.makedirs(EVAL_DIR, exist_ok=True)
    report_path = os.path.join(EVAL_DIR, "report.txt")

    joints = list(JOINT_MAP.keys())
    W = 65

    lines = []
    lines.append("=" * W)
    lines.append("   POSE ENGINE - ACCURACY & PRECISION EVALUATION REPORT")
    lines.append("=" * W)
    lines.append("  Dataset         : COCO 2017 Val (single-person, 6 joints visible)")
    lines.append("  Images evaluated: %d" % n_images)
    lines.append("  Detection fails : %d  (%.1f%%)" % (n_failed, n_failed / max(n_images, 1) * 100))
    lines.append("  Time taken      : %.1fs" % elapsed)
    lines.append("")
    lines.append("-" * W)
    lines.append("  ACCURACY  -  PCK (Percentage of Correct Keypoints)")
    lines.append("  A joint is 'correct' if prediction is within threshold of GT")
    lines.append("-" * W)

    header = "  %-22s %10s %10s %10s" % ("Joint", "PCK@10px", "PCK@20px", "PCK@50px")
    lines.append(header)
    lines.append("  " + "-" * 22 + " " + "-" * 10 + " " + "-" * 10 + " " + "-" * 10)

    for joint in joints:
        row = "  %-22s" % joint
        for t in PCK_THRESHOLDS:
            row += " %9.1f%%" % pck[t].get(joint, 0)
        lines.append(row)

    lines.append("  " + "-" * 22 + " " + "-" * 10 + " " + "-" * 10 + " " + "-" * 10)
    overall_row = "  %-22s" % "OVERALL"
    for t in PCK_THRESHOLDS:
        overall_row += " %9.1f%%" % pck[t].get("overall", 0)
    lines.append(overall_row)

    lines.append("")
    lines.append("-" * W)
    lines.append("  PRECISION  -  Error Standard Deviation per Joint")
    lines.append("  Lower StdDev = More consistent / stable predictions")
    lines.append("-" * W)
    lines.append("  %-22s %12s %12s %10s" % ("Joint", "Mean Error", "Std Dev (+-)", "Rating"))
    lines.append("  " + "-" * 22 + " " + "-" * 12 + " " + "-" * 12 + " " + "-" * 10)

    for joint in joints:
        me = mean_err.get(joint, 0)
        sd = std_err.get(joint, 0)
        if sd < 5:
            rating = "Excellent"
        elif sd < 15:
            rating = "Good"
        elif sd < 30:
            rating = "Fair"
        else:
            rating = "Poor"
        lines.append("  %-22s %10.1fpx %10.1fpx %10s" % (joint, me, sd, rating))

    lines.append("")
    lines.append("-" * W)
    lines.append("  INTERPRETATION GUIDE")
    lines.append("-" * W)
    lines.append("  PCK@10px > 80%  -> Highly accurate (sub-10px error)")
    lines.append("  PCK@10px > 60%  -> Good accuracy")
    lines.append("  PCK@10px < 40%  -> Needs improvement")
    lines.append("  StdDev < 5px    -> Excellent precision (very stable)")
    lines.append("  StdDev < 15px   -> Good precision")
    lines.append("  StdDev > 30px   -> Poor precision (high jitter)")
    lines.append("")
    lines.append("=" * W)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text, report_path


def save_csv(rows, header):
    os.makedirs(EVAL_DIR, exist_ok=True)
    csv_path = os.path.join(EVAL_DIR, "per_joint.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return csv_path


# =============================================================================
#   MAIN EVALUATION LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate PoseEngine accuracy & precision against COCO.")
    parser.add_argument("--max-images",    type=int, default=300,
                        help="Max COCO images to evaluate (default: 300)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download and use already-present COCO files")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("   POSE ENGINE - ACCURACY & PRECISION EVALUATOR")
    print("=" * 65)
    print("\n  Mode: Evaluating up to %d COCO Val 2017 images\n" % args.max_images)

    print("[1/4] Checking COCO dataset ...")
    ensure_coco_data(skip_download=args.skip_download)

    print("\n[2/4] Loading annotations ...")
    samples = load_coco_annotations(max_images=args.max_images)

    if not samples:
        print("\n[ERROR] No usable COCO images found. Check the data directory.")
        sys.exit(1)

    print("\n[3/4] Running PoseEngine on images ...")
    from pose_engine import PoseEngine
    engine = PoseEngine()

    joints = list(JOINT_MAP.keys())
    errors_by_joint = {j: [] for j in joints}
    csv_rows = []
    csv_header = (["image_id", "file_name"] +
                  [j + "_gt_x"    for j in joints] +
                  [j + "_pred_x"  for j in joints] +
                  [j + "_error_px" for j in joints] +
                  ["detected"])

    n_failed = 0
    t_start  = time.time()

    for i, sample in enumerate(samples):
        img_path = os.path.join(IMAGES_DIR, sample["file_name"])
        if not os.path.isfile(img_path):
            n_failed += 1
            continue

        sys.stdout.write("\r  [%4d/%d]  %-40s" % (i + 1, len(samples), sample["file_name"][:40]))
        sys.stdout.flush()

        result = engine.get_keypoints(img_path, skip_boundaries=True)

        row = [sample["image_id"], sample["file_name"]]

        for j in joints:
            gt = sample["gt_joints"][j]
            row.append(gt[0])

        if isinstance(result, str):
            n_failed += 1
            for j in joints:
                row.append("N/A")
            for j in joints:
                row.append("N/A")
            row.append("NO")
            csv_rows.append(row)
            continue

        for j in joints:
            pred = result.get(j)
            row.append(pred[0] if pred else "N/A")

        for j in joints:
            gt   = sample["gt_joints"][j]
            pred = result.get(j)
            if pred is None:
                row.append("N/A")
                n_failed += 1
            else:
                err = euclidean(pred, gt)
                errors_by_joint[j].append(err)
                row.append(round(err, 2))

        row.append("YES")
        csv_rows.append(row)

    print("")

    elapsed    = time.time() - t_start

    print("\n[4/4] Computing metrics and saving report ...")
    pck, mean_err, std_err = compute_metrics(errors_by_joint)
    report_text, report_path = save_report(pck, mean_err, std_err, len(samples), n_failed, elapsed)
    csv_path = save_csv(csv_rows, csv_header)

    print("\n" + report_text)
    print("\n  [OK] Full report saved to : " + os.path.abspath(report_path))
    print("  [OK] Per-joint CSV saved to: " + os.path.abspath(csv_path))
    print("")


if __name__ == "__main__":
    main()
