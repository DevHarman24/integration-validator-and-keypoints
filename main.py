from pose_engine import PoseEngine
import json
import os

import sys

def main():
    engine = PoseEngine()
    
    # Check if images were passed as arguments
    if len(sys.argv) > 1:
        images_to_process = sys.argv[1:]
    else:
        # Default fallback
        images_to_process = ["front.jpg", "side.jpg"]
    
    any_processed = False

    for img_name in images_to_process:
        if os.path.exists(img_name):
            print(f"Processing {img_name}...")
            keypoints = engine.get_keypoints(img_name)
            
            if isinstance(keypoints, str):
                print(f"Error in {img_name}: {keypoints}")
            else:
                print(f"Keypoints for {img_name} extracted successfully:")
                print(json.dumps(keypoints, indent=4))
                
                # Create visualization name (e.g., front_visualized.jpg)
                base, ext = os.path.splitext(img_name)
                vis_name = f"{base}_visualized{ext}"
                engine.visualize_keypoints(img_name, vis_name, show=False)
                print(f"Visualization saved to '{vis_name}'")
            
            any_processed = True
        else:
            if len(sys.argv) > 1:
                print(f"Error: File '{img_name}' not found.")
            else:
                print(f"Info: Default file '{img_name}' not found. Skipping.")

    if not any_processed:
        print("\n[!] No images processed.")
        print("Usage: python main.py <image1.jpg> <image2.jpg> ...")
        print("Or place 'front.jpg' or 'side.jpg' in the directory for auto-processing.")

if __name__ == "__main__":
    main()
