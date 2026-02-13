"""
Quick Detection Script - Simple interface for live detection
"""

import sys
import os

def main():
    print("=" * 60)
    print("Live Waste Classification Detection")
    print("=" * 60)
    print("\nChoose detection mode:")
    print("1. Single Image Detection")
    print("2. Webcam Live Feed")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        image_path = input("\nEnter image path: ").strip().strip('"')
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            return
        
        print(f"\nRunning detection on: {image_path}")
        os.system(f'python live_detect.py --image "{image_path}"')
    
    elif choice == "2":
        print("\nStarting webcam feed...")
        print("Press 'Q' to quit, 'S' to save frame")
        os.system('python live_detect.py --webcam')
    
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

