"""
Simple Launcher for Live Detection
Easy-to-use interface for starting live waste detection
"""

import os
import sys
from pathlib import Path

def check_model_exists():
    """Check if model file exists"""
    model_path = Path('models/best_model.pth')
    if not model_path.exists():
        print("=" * 60)
        print("ERROR: Model not found!")
        print("=" * 60)
        print(f"Expected model at: {model_path}")
        print("\nPlease train the model first:")
        print("  python train_efficientdet.py --data D:/garbage_detection/Try/dataset")
        return False
    return True

def main():
    print("=" * 60)
    print("Waste Classification - Live Detection")
    print("=" * 60)
    
    if not check_model_exists():
        sys.exit(1)
    
    print("\nChoose detection mode:")
    print("1. Webcam Live Detection (Recommended)")
    print("2. Single Image Detection")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\nStarting webcam detection...")
        print("Controls:")
        print("  - Press 'Q' to quit")
        print("  - Press 'S' or 'C' to save current frame")
        print("\n" + "=" * 60)
        os.system('python live_detection.py --webcam')
    
    elif choice == "2":
        image_path = input("\nEnter image path: ").strip().strip('"')
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            return
        
        print(f"\nRunning detection on: {image_path}")
        print("=" * 60)
        os.system(f'python live_detection.py --image "{image_path}"')
    
    elif choice == "3":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")

