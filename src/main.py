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
        # Default fallback - look in data/samples
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
            print(f"Processing {img_name}...")
            keypoints = engine.get_keypoints(img_path)
            
            if isinstance(keypoints, str):
                print(f"Error in {img_name}: {keypoints}")
            else:
                print(f"Keypoints for {img_name} extracted successfully:")
                print(json.dumps(keypoints, indent=4))
                
                # Create visualization name in output dir
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
