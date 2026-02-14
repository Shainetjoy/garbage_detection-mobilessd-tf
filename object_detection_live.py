"""
Live Object Detection for Waste Classification
Detects objects in frame and only shows detected waste items with bounding boxes
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json
import argparse
from pathlib import Path
import time
import numpy as np

class ObjectDetector:
    def __init__(self, model_path, class_info_path=None, device='cuda'):
        """Initialize the detector with trained model"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print("=" * 60)
        print("Loading Waste Classification Model")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Device: {self.device}")
        
        # Load class information
        if class_info_path is None:
            class_info_path = Path(model_path).parent / 'class_info.json'
        
        if Path(class_info_path).exists():
            with open(class_info_path, 'r') as f:
                class_info = json.load(f)
            self.classes = class_info['classes']
            self.idx_to_class = {int(k): v for k, v in class_info['idx_to_class'].items()}
        else:
            print("Warning: class_info.json not found. Using default classes.")
            self.classes = ['cardboard', 'clothes', 'glass', 'paper', 'plastic', 'shoes', 'trash']
            self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        
        # Load model
        import torchvision
        model = torchvision.models.efficientnet_b0(pretrained=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(model.classifier[1].in_features, len(self.classes))
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Background subtractor for object detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        print(f"Classes: {', '.join(self.classes)}")
        print("=" * 60)
        print("Model loaded successfully!")
        print("=" * 60)
    
    def detect_objects(self, frame):
        """Detect objects in frame using background subtraction"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 500  # Minimum object area
        detected_objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Expand bounding box slightly
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                detected_objects.append((x, y, w, h))
        
        return detected_objects
    
    def classify_object(self, frame, bbox):
        """Classify a detected object"""
        x, y, w, h = bbox
        
        # Extract object region
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None, 0.0, None
        
        # Convert to RGB and PIL Image
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(roi_rgb)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            predicted_class = self.idx_to_class[predicted.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score, probabilities
    
    def detect_webcam(self, camera_index=0, conf_threshold=0.5, skip_frames=5, min_conf=0.70, max_conf=0.85):
        """Live object detection from webcam"""
        print("\n" + "=" * 60)
        print("Starting Live Object Detection")
        print("=" * 60)
        print(f"Camera: {camera_index}")
        print(f"Confidence Threshold: {conf_threshold:.0%}")
        print(f"Display Range: {min_conf:.0%} - {max_conf:.0%} (only show boxes in this range)")
        print(f"Frame Skip: Processing every {skip_frames} frames")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Save current frame")
        print("  +/- - Adjust frame skip")
        print("=" * 60)
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        save_path = Path('predictions')
        save_path.mkdir(exist_ok=True)
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        detected_items = []  # Store detected objects with their classifications
        
        print("\nLive detection started! Move objects in front of camera.\n")
        print("Note: Background subtraction needs a few seconds to initialize.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Only detect objects every N frames
                if frame_count % skip_frames == 0:
                    # Detect objects in frame
                    objects = self.detect_objects(frame)
                    detected_items = []
                    
                    # Classify each detected object
                    for bbox in objects:
                        predicted_class, confidence, probabilities = self.classify_object(frame, bbox)
                        
                        # Only show if confidence is between min_conf and max_conf
                        if predicted_class and confidence >= conf_threshold:
                            if min_conf <= confidence <= max_conf:
                                detected_items.append({
                                    'bbox': bbox,
                                    'class': predicted_class,
                                    'confidence': confidence
                                })
                
                # Draw only detected objects
                output_frame = frame.copy()
                
                if detected_items:
                    for item in detected_items:
                        x, y, w, h = item['bbox']
                        class_name = item['class']
                        confidence = item['confidence']
                        
                        # Choose color based on class
                        colors = {
                            'cardboard': (0, 165, 255),  # Orange
                            'clothes': (255, 0, 255),    # Magenta
                            'glass': (255, 255, 0),      # Cyan
                            'paper': (0, 255, 255),      # Yellow
                            'plastic': (255, 0, 0),      # Blue
                            'shoes': (0, 255, 0),       # Green
                            'trash': (0, 0, 255)        # Red
                        }
                        color = colors.get(class_name, (255, 255, 255))
                        
                        # Draw bounding box
                        cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.1%}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        label_y = max(y - 10, label_size[1] + 10)
                        
                        # Background for text
                        cv2.rectangle(output_frame, 
                                    (x, label_y - label_size[1] - 5),
                                    (x + label_size[0] + 5, label_y + 5),
                                    color, -1)
                        
                        # Text
                        cv2.putText(output_frame, label, (x + 2, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add info overlay
                h, w = output_frame.shape[:2]
                overlay = output_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)
                
                # FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                # Info text
                info_text = f"Detected Objects: {len(detected_items)} | FPS: {fps:.1f}"
                cv2.putText(output_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                skip_text = f"Frame Skip: {skip_frames} (Press +/-)"
                cv2.putText(output_frame, skip_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.putText(output_frame, "Press 'Q' to quit | 'S' to save",
                           (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Live Object Detection - Waste Classification', output_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    filepath = save_path / filename
                    cv2.imwrite(str(filepath), output_frame)
                    print(f"Frame saved: {filepath}")
                elif key == ord('+') or key == ord('='):
                    skip_frames = min(30, skip_frames + 1)
                    print(f"Frame skip: {skip_frames}")
                elif key == ord('-') or key == ord('_'):
                    skip_frames = max(1, skip_frames - 1)
                    print(f"Frame skip: {skip_frames}")
        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped.")

def main():
    parser = argparse.ArgumentParser(description='Live Object Detection for Waste Classification')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model file')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for live detection')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Minimum confidence threshold for detection (default: 0.5)')
    parser.add_argument('--min-conf', type=float, default=0.70,
                       help='Minimum confidence to display bounding box (default: 0.70)')
    parser.add_argument('--max-conf', type=float, default=0.85,
                       help='Maximum confidence to display bounding box (default: 0.85)')
    parser.add_argument('--skip', type=int, default=5,
                       help='Process every N frames (default: 5)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    try:
        detector = ObjectDetector(args.model, device=args.device)
        
        if args.webcam:
            detector.detect_webcam(args.camera, args.conf, args.skip, args.min_conf, args.max_conf)
        else:
            print("\n" + "=" * 60)
            print("Live Object Detection for Waste Classification")
            print("=" * 60)
            print("\nUsage:")
            print("  python object_detection_live.py --webcam")
            print("\nOptions:")
            print("  --camera N       Camera index (default: 0)")
            print("  --conf X         Minimum confidence for detection (default: 0.5)")
            print("  --min-conf X     Minimum confidence to display box (default: 0.70)")
            print("  --max-conf X     Maximum confidence to display box (default: 0.85)")
            print("  --skip N         Process every N frames (default: 5)")
            print("\nExample:")
            print("  python object_detection_live.py --webcam")
            print("  python object_detection_live.py --webcam --min-conf 0.70 --max-conf 0.85")
            parser.print_help()
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease make sure the model exists at: models/best_model.pth")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

