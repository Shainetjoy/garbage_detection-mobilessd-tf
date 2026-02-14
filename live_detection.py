"""
Enhanced Live Detection for Waste Classification
Real-time webcam detection with improved UI and features
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

class LiveWasteDetector:
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
        
        print(f"Classes: {', '.join(self.classes)}")
        print("=" * 60)
        print("Model loaded successfully!")
        print("=" * 60)
    
    def predict_frame(self, frame):
        """Predict on a single frame"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
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
    
    def draw_prediction(self, frame, predicted_class, confidence_score, probabilities, fps=0):
        """Draw prediction results on frame"""
        h, w = frame.shape[:2]
        
        # Create overlay for better visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Main prediction
        if confidence_score >= 0.5:
            color = (0, 255, 0)  # Green
            status = "HIGH CONFIDENCE"
        elif confidence_score >= 0.3:
            color = (0, 165, 255)  # Orange
            status = "MEDIUM CONFIDENCE"
        else:
            color = (0, 0, 255)  # Red
            status = "LOW CONFIDENCE"
        
        # Main prediction text
        main_text = f"{predicted_class.upper()}: {confidence_score:.1%}"
        cv2.putText(frame, main_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Status
        cv2.putText(frame, status, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, min(3, len(self.classes)))
        y_offset = 110
        cv2.putText(frame, "Top Predictions:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            cls_name = self.idx_to_class[idx.item()]
            prob_val = prob.item()
            bar_width = int(prob_val * 200)
            
            # Prediction text
            text = f"{i+1}. {cls_name}: {prob_val:.1%}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Confidence bar
            cv2.rectangle(frame, (200, y_offset - 15), (200 + bar_width, y_offset - 5),
                         (0, 255, 255), -1)
            cv2.rectangle(frame, (200, y_offset - 15), (400, y_offset - 5),
                         (255, 255, 255), 1)
            
            y_offset += 30
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit | 'S' to save | 'C' to capture", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def detect_webcam(self, camera_index=0, conf_threshold=0.3, save_dir='predictions', skip_frames=5):
        """Live detection from webcam with frame skipping for better performance"""
        print("\n" + "=" * 60)
        print("Starting Live Detection")
        print("=" * 60)
        print(f"Camera: {camera_index}")
        print(f"Confidence Threshold: {conf_threshold:.0%}")
        print(f"Frame Skip: Processing every {skip_frames} frames")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Save current frame")
        print("  C - Capture and save")
        print("  +/- - Increase/Decrease frame skip (1-30)")
        print("=" * 60)
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        inference_times = []
        
        # Store last prediction to display on skipped frames
        last_predicted_class = "Waiting..."
        last_confidence_score = 0.0
        last_probabilities = None
        
        print("\nLive detection started! Press 'Q' to quit.\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Only predict every N frames
                if frame_count % skip_frames == 0:
                    # Predict on this frame
                    inference_start = time.time()
                    predicted_class, confidence_score, probabilities = self.predict_frame(frame)
                    inference_time = (time.time() - inference_start) * 1000  # in ms
                    inference_times.append(inference_time)
                    
                    # Update last prediction
                    last_predicted_class = predicted_class
                    last_confidence_score = confidence_score
                    last_probabilities = probabilities
                    
                    # Calculate FPS (average over last 30 frames)
                    if len(inference_times) >= 30:
                        fps = 30 / (time.time() - fps_start_time)
                        fps_start_time = time.time()
                        avg_inference = np.mean(inference_times[-30:])
                        print(f"FPS: {fps:.1f} | Avg Inference: {avg_inference:.1f}ms | "
                              f"Current: {predicted_class} ({confidence_score:.1%}) | Skip: {skip_frames}")
                
                # Draw last prediction on current frame (even if we skipped detection)
                frame = self.draw_prediction(frame, last_predicted_class, last_confidence_score, 
                                            last_probabilities if last_probabilities is not None else 
                                            torch.zeros(len(self.classes)), fps)
                
                # Add frame skip indicator
                h, w = frame.shape[:2]
                skip_text = f"Frame Skip: {skip_frames} (Press +/- to adjust)"
                cv2.putText(frame, skip_text, (10, h - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Show frame
                cv2.imshow('Live Waste Classification', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S') or key == ord('c') or key == ord('C'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{last_predicted_class}_{last_confidence_score:.2f}_{timestamp}.jpg"
                    filepath = save_path / filename
                    cv2.imwrite(str(filepath), frame)
                    print(f"Frame saved: {filepath}")
                elif key == ord('+') or key == ord('='):
                    # Increase frame skip
                    skip_frames = min(30, skip_frames + 1)
                    print(f"Frame skip increased to: {skip_frames}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease frame skip
                    skip_frames = max(1, skip_frames - 1)
                    print(f"Frame skip decreased to: {skip_frames}")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nStopping live detection...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            if inference_times:
                avg_inference = np.mean(inference_times)
                print(f"\nAverage inference time: {avg_inference:.2f} ms")
                print(f"Average FPS: {1000/avg_inference:.1f}")
                print(f"Final frame skip: {skip_frames}")
            print("Live detection stopped.")
    
    def detect_image(self, image_path, show=True, save=True):
        """Detect on a single image"""
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            return None
        
        print(f"\nProcessing image: {image_path}")
        
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not load image: {image_path}")
            return None
        
        # Predict
        inference_start = time.time()
        predicted_class, confidence_score, probabilities = self.predict_frame(frame)
        inference_time = (time.time() - inference_start) * 1000
        
        # Draw predictions
        frame = self.draw_prediction(frame, predicted_class, confidence_score, 
                                    probabilities, fps=1000/inference_time)
        
        # Print results
        print("\n" + "=" * 60)
        print("Prediction Results:")
        print("=" * 60)
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence_score:.2%}")
        print(f"Inference Time: {inference_time:.2f} ms")
        print("\nAll Class Probabilities:")
        for idx, cls in self.idx_to_class.items():
            prob = probabilities[idx].item()
            marker = " <--" if cls == predicted_class else ""
            print(f"  {cls}: {prob:.2%}{marker}")
        print("=" * 60)
        
        # Display image
        if show:
            cv2.imshow('Waste Classification - Press Q to close', frame)
            print("\nPress 'Q' to close the image window...")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or cv2.getWindowProperty('Waste Classification - Press Q to close', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()
        
        # Save annotated image
        if save:
            output_dir = Path('predictions')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"predicted_{Path(image_path).name}"
            cv2.imwrite(str(output_path), frame)
            print(f"\nAnnotated image saved to: {output_path}")
        
        return {
            'class': predicted_class,
            'confidence': confidence_score,
            'inference_time': inference_time
        }

def main():
    parser = argparse.ArgumentParser(description='Live Waste Classification Detection')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image file')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for live detection')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--skip', type=int, default=5,
                       help='Process every N frames (default: 5, higher = faster)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display images')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save prediction results')
    
    args = parser.parse_args()
    
    try:
        detector = LiveWasteDetector(args.model, device=args.device)
        
        if args.webcam:
            detector.detect_webcam(args.camera, args.conf, skip_frames=args.skip)
        elif args.image:
            detector.detect_image(
                args.image,
                show=not args.no_show,
                save=not args.no_save
            )
        else:
            print("\n" + "=" * 60)
            print("Live Waste Classification Detection")
            print("=" * 60)
            print("\nUsage:")
            print("  Webcam:  python live_detection.py --webcam")
            print("  Image:   python live_detection.py --image path/to/image.jpg")
            print("\nOptions:")
            print("  --camera N    Camera index (default: 0)")
            print("  --conf X      Confidence threshold (default: 0.3)")
            print("  --skip N      Process every N frames (default: 5, higher = faster)")
            print("  --device      cuda or cpu (default: auto)")
            print("\nExample:")
            print("  python live_detection.py --webcam")
            print("  python live_detection.py --webcam --skip 10  # Process every 10 frames")
            print("  python live_detection.py --image test.jpg --conf 0.5")
            parser.print_help()
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease make sure:")
        print("  1. The model has been trained (run: python train_efficientdet.py)")
        print("  2. The model path is correct")
        print("  3. Model file exists at: models/best_model.pth")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

